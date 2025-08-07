"""
Context Optimization (CoOp) Trainer Implementation.

This module implements the CoOp (Context Optimization) method for few-shot learning
with CLIP models. CoOp learns continuous prompt tokens that can be optimized
end-to-end for downstream tasks.

Reference: "Learning to Prompt for Vision-Language Models" (ICCV 2021)
"""

import os.path as osp
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg) -> nn.Module:
    """
    Load CLIP model to CPU for initialization.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        nn.Module: Loaded CLIP model
    """
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # Loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    design_details = {
        "trainer": 'CoOp',
        "vision_depth": 0,
        "language_depth": 0, 
        "vision_ctx": 0,
        "language_ctx": 0
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    """
    Text encoder module for processing prompt tokens.
    
    This module handles the text encoding part of CLIP, processing
    learned prompt tokens through the transformer architecture.
    """
    
    def __init__(self, clip_model: nn.Module):
        """
        Initialize text encoder with CLIP model components.
        
        Args:
            clip_model: Pre-trained CLIP model
        """
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            prompts: Learned prompt tokens [batch_size, n_ctx, ctx_dim]
            tokenized_prompts: Tokenized text prompts [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Text features [batch_size, feature_dim]
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # Extract features from the EOT embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    """
    Prompt learning module for CoOp.
    
    This module learns continuous prompt tokens that can be optimized
    end-to-end for downstream classification tasks.
    """
    
    def __init__(self, cfg, classnames: List[str], clip_model: nn.Module):
        """
        Initialize prompt learner.
        
        Args:
            cfg: Configuration object
            classnames: List of class names
            clip_model: Pre-trained CLIP model
        """
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # To be optimized
        self.meta_net = None
        self.label_ctx = None
        
        # Class-specific context (CSC) initialization
        if cfg.TRAINER.COOP.CSC:
            # Use a simple network to generate the ctx_vectors
            ctx_dim = clip_model.ln_final.weight.shape[0]
            self.meta_net = nn.Sequential(
                nn.Linear(ctx_dim, ctx_dim // 16),
                nn.ReLU(inplace=True),
                nn.Linear(ctx_dim // 16, ctx_dim)
            )
            
            # Class-specific context: ctx_vectors generated per class
            if cfg.TRAINER.COOP.CTX_INIT:
                # Use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors = ctx_vectors.unsqueeze(0).expand(n_cls, -1, -1)
            else:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            self.label_ctx = nn.Parameter(ctx_vectors)  # To be optimized

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.COOP.CSC
        self.classnames = classnames

    def construct_prompts(self, ctx: torch.Tensor, prefix: str, suffix: str, label: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct prompts for text encoding.
        
        Args:
            ctx: Context tokens [dim0, n_ctx, ctx_dim]
            prefix: Prefix tokens [n_cls, 1, ctx_dim]
            suffix: Suffix tokens [n_cls, *, ctx_dim]
            label: Class label (optional)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prompts and tokenized prompts
        """
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.csc:
            _ctx = ctx[label, :, :] if label is not None else ctx
        else:
            _ctx = ctx

        prompts = []
        for i in range(self.n_cls):
            ctx_i = _ctx[i:i+1, :, :]
            prefix_i = prefix[i:i+1, :, :]
            suffix_i = suffix[i:i+1, :, :]
            prompt_i = torch.cat([prefix_i, ctx_i, suffix_i], dim=1)
            prompts.append(prompt_i)
        prompts = torch.cat(prompts, dim=0)

        return prompts

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to generate prompts.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prompts and tokenized prompts
        """
        if self.meta_net is not None:
            ctx = self.meta_net(self.label_ctx)
        else:
            ctx = self.ctx

        if self.csc:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = "a photo of a"
        suffix = "."
        prompts = self.construct_prompts(ctx, prefix, suffix)
        tokenized_prompts = clip.tokenize(prompts)

        return prompts, tokenized_prompts


class CustomCLIP(nn.Module):
    """
    Custom CLIP model with prompt learning capabilities.
    
    This module combines the visual encoder from CLIP with learned
    text prompts for few-shot classification.
    """
    
    def __init__(self, cfg, classnames: List[str], clip_model: nn.Module):
        """
        Initialize custom CLIP model.
        
        Args:
            cfg: Configuration object
            classnames: List of class names
            clip_model: Pre-trained CLIP model
        """
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through custom CLIP model.
        
        Args:
            image: Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """
    Context Optimization (CoOp) trainer.
    
    This trainer implements the CoOp method for few-shot learning with CLIP models.
    It learns continuous prompt tokens that can be optimized end-to-end.
    """
    
    def check_cfg(self, cfg):
        """Check configuration for CoOp trainer."""
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """Build the CoOp model."""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in the image encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow since CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """Forward and backward pass for training."""
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.scaler.scale(loss).backward()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            loss.backward()

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """Parse batch for training."""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """Load model from checkpoint."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is named after the dataset with the best performance
        if epoch is None:
            filepath = osp.join(directory, "model-best.pth.tar")
        else:
            filepath = osp.join(directory, f"model-epoch{epoch}.pth.tar")

        if not osp.exists(filepath):
            return

        print(f"Loading checkpoint from {filepath}")
        checkpoint = load_checkpoint(filepath, map_location="cpu")

        for name in names:
            model = self.models[name]
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint[name], strict=False
            )
            if missing_keys:
                print(f"Missing keys in {name}: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in {name}: {unexpected_keys}")

        # When in eval mode, print the model structure
        if not self.training:
            print("Note that the model structure is printed when in eval mode")
            for name in names:
                model = self.models[name]
                print(f"Model {name}:")
                print(model)
