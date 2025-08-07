"""
Multi-modal Adaptive Prompt Learning (MaPLe) Trainer Implementation.

This module implements the MaPLe method for few-shot learning with CLIP models.
MaPLe learns both shallow and deep prompts for both vision and language modalities,
enabling better adaptation to downstream tasks.

Reference: "MaPLe: Multi-modal Prompt Learning" (CVPR 2023)
"""

import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Union

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import QuickGELU
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.Adapter import Adapter2Linear

from oriCLIP import clip as oriCLIP
import random
import sys
import time
import numpy as np
import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

_tokenizer = _Tokenizer()




def load_clip_to_cpu(cfg) -> nn.Module:
    """
    Load CLIP model to CPU for initialization.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        nn.Module: Loaded CLIP model with MaPLe design details
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
        "trainer": 'MaPLe',
        "vision_depth": 0,
        "language_depth": 0, 
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
        "SAP_depth": cfg.TRAINER.MAPLE.MEMORY_DEPTH,
        "text_range": range(0, 11),
        "image_range": range(0, 11)
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    """
    Enhanced text encoder module for MaPLe with compound prompts support.
    
    This module handles the text encoding part of CLIP with support for
    compound prompts at deeper layers and memory mechanisms.
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

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor, 
                compound_prompts_deeper_text: Optional[torch.Tensor] = None,
                ks: Optional[torch.Tensor] = None, 
                vs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through text encoder with compound prompts support.
        
        Args:
            prompts: Learned prompt tokens [batch_size, n_ctx, ctx_dim]
            tokenized_prompts: Tokenized text prompts [batch_size, seq_len]
            compound_prompts_deeper_text: Compound prompts for deeper layers
            ks: Key vectors for memory mechanism
            vs: Value vectors for memory mechanism
            
        Returns:
            torch.Tensor: Text features [batch_size, feature_dim]
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0, ks, vs]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Extract features from the EOT embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    """
    Multi-modal prompt learning module for MaPLe.
    
    This module learns both shallow and deep prompts for both vision and language
    modalities, enabling better adaptation to downstream tasks.
    """
    
    def __init__(self, cfg, classnames: List[str], clip_model: nn.Module):
        """
        Initialize multi-modal prompt learner.
        
        Args:
            cfg: Configuration object
            classnames: List of class names
            clip_model: Pre-trained CLIP model
        """
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()

        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_img = nn.Parameter(torch.HalfTensor(3, 768))
        nn.init.normal_(self.ctx_img, std=0.02)






        # Class names processing
        self.classnames = classnames
        self.class_numbers = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens


    def construct_prompts(self, ctx: torch.Tensor, prefix: torch.Tensor, 
                         suffix: torch.Tensor, label: Optional[int] = None) -> torch.Tensor:
        """
        Construct prompts for text encoding.
        
        Args:
            ctx: Context tokens [dim0, n_ctx, ctx_dim]
            prefix: Prefix tokens [n_cls, 1, ctx_dim]
            suffix: Suffix tokens [n_cls, *, ctx_dim]
            label: Class label (optional)
            
        Returns:
            torch.Tensor: Constructed prompts
        """
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """
        Forward pass to generate prompts for both modalities.
        
        Returns:
            Tuple containing:
                - prompts: Text prompts
                - ctx_img: Visual prompts
                - compound_prompts_text: Compound text prompts (empty for now)
                - visual_deep_prompts: Visual deep prompts (empty for now)
        """
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform prompts to 768 for the visual side
        visual_deep_prompts = []

        return prompts, self.ctx_img, [], visual_deep_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.swiglu = SwiGLU()
        self.swiglu = nn.GELU()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.Memory_length = self.cfg.TRAINER.MAPLE.MEMORY_LENGTH
        self.m_text = 4

        # self.text_layer_depth = 12
        self.m_img = 4
        self.design_details = clip_model.design_details

        # self.Memory_depth = self.design_details["image_range"]
        self.text_layer_depth = self.design_details["text_range"]
        self.img_layer_depth =  self.design_details["image_range"]
        self.h = 8
        self.h_img = 12
        self.d_k = 64
        self.d_v = 64
        self.depth = 11
        # single_layer = nn.Linear(512, 768).half()
        # self.adapter_projections = _get_clones(single_layer, self.layer_depth)
        # self.single_adapter_k = Adapter2Linear(512,768).half()
        # self.single_k_adapter = nn.Linear(512,768).half()
        # self.single_v_adapter = nn.Linear(512,768).half()
        # self.single_k_adapter = Adapter2Linear(512, 768).half()
        # self.single_v_adapter = Adapter2Linear(512, 768).half()
        # self.k_adapter_weight = nn.Parameter(torch.HalfTensor(1,1).to(device)).half()
        # self.v_adapter_weight = nn.Parameter(torch.HalfTensor(1, 1).to(device)).half()


        # self.k_adapter = nn.ParameterList([nn.Parameter(torch.Tensor(1,self.m_text, self.h * self.d_k)) #if i in self.text_layer_depth else None
        #                                                for i in range(11)])
        # self.v_adapter = nn.ParameterList([nn.Parameter(torch.Tensor(1, self.m_text, self.h * self.d_k)) #if i in self.text_layer_depth else None
        #                                                for i in range(11)])

        # self.k_adapter = nn.ParameterList(
        #     [nn.Parameter(torch.Tensor(1, self.m_text, self.h * self.d_k))  # if i in self.text_layer_depth else None
        #      for i in range(11)])
        # self.v_adapter = nn.ParameterList(
        #     [nn.Parameter(torch.Tensor(1, self.m_text, self.h * self.d_k))  # if i in self.text_layer_depth else None
        #      for i in range(11)])
        self.k_adapter = nn.ParameterList([nn.Parameter(torch.Tensor(1,self.m_text, self.h * self.d_k)) for _ in range(11)])  # if i in self.text_layer_depth else None
        self.v_adapter = nn.ParameterList([nn.Parameter(torch.Tensor(1,self.m_text, self.h * self.d_k)) for _ in range(11)])
        self.k_adapter_img = nn.ParameterList([nn.Parameter(torch.Tensor(1, self.m_img, self.h_img * self.d_v)) for _ in range(11)]) #if i in self.img_layer_depth else None
        self.v_adapter_img = nn.ParameterList([nn.Parameter(torch.Tensor(1, self.m_img, self.h_img * self.d_v)) for _ in range(11)])





        # self.weight_photo_adapter = nn.Parameter(torch.HalfTensor(1, 1))

        # self.dropout_prompt = nn.Dropout(p=0.6)
        self.dropout_prompt = nn.Dropout(p=0.6)
        # self.linear_adapter  = None
        # if self.prompt_learner.training:
        #     self.linear_adapter = nn.Linear(512,5)
        # self.dropout_linear = nn.Dropout(p=0.2)
        #self.adapter_attn = nn.MultiheadAttention(self.h * self.d_k,2,batch_first=True).half()
        # self.adapter_linear = nn.Linear(self.h * self.d_k,self.h * self.d_k).half()
        # self.img_adapter = nn.Parameter(torch.HalfTensor(1, 512))

        # self.text_adapter = nn.Parameter(torch.HalfTensor(1, 512))
        # self.text2_adapter = nn.Linear(512,512).half()

        self.initWeight()



    def initWeight(self):

        # std = 0.0001
        # const_value = 0.01
        # nn.init.normal_(self.key_img,std=0.02)
        # self.key_img = nn.Parameter(self.key_img.half())
        # nn.init.normal_(self.k_adapter, 0, 1 / (self.d_k))
        # nn.init.normal_(self.v_adapter, 0, 1 / (self.d_k))
        # nn.init.normal_(self.k_adapter_img, 0, 1 / (self.d_k))
        # nn.init.normal_(self.v_adapter_img, 0, 1 / (self.d_k))
        for i in self.k_adapter:
            # nn.init.zeros_(i)
            # nn.init.normal_(i, 0,std=1/self.d_k)
            if i != None:
                # nn.init.zeros_(i)
                # nn.init.trunc_normal_(i)
                nn.init.normal_(i, 0,1/(self.d_k))
                # nn.init.xavier_uniform_(i)
            # nn.init.constant_(i,const_value)
            #     nn.init.kaiming_normal_(i)
            # torch.nn.init.uniform_(i)
            # torch.nn.init.normal_(i,0,std)
        for i in self.v_adapter:
            # nn.init.zeros_(i)
            # nn.init.zeros_(i)
            # nn.init.normal_(i, std=1/self.d_k)
            if i != None:
                # nn.init.zeros_(i)
                # nn.init.trunc_normal_(i)
                nn.init.normal_(i, 0, 1/(self.d_k))
            # nn.init.constant_(i, const_value)
            #     nn.init.xavier_uniform_(i)
            #     nn.init.kaiming_normal_(i)
            # torch.nn.init.uniform_(i)
            # torch.nn.init.normal_(i, 0, std)
        for i in self.k_adapter_img:
            # nn.init.zeros_(i)
            # nn.init.normal_(i, std=1/self.d_k)
            if i != None:
                # nn.init.zeros_(i)
                # nn.init.trunc_normal_(i)
                nn.init.normal_(i, 0, 1/(self.d_k))
            # nn.init.constant_(i, const_value)
            #     nn.init.xavier_uniform_(i)
            #     nn.init.kaiming_normal_(i)
            # torch.nn.init.uniform_(i)
            # torch.nn.init.normal_(i, 0, std)

        for i in self.v_adapter_img:
            # nn.init.zeros_(i)
            # nn.init.normal_(i, std=1/self.d_k)
            if i != None:
                # nn.init.zeros_(i)
                # nn.init.trunc_normal_(i)
                nn.init.normal_(i, 0, 1/(self.d_k))
            # nn.init.constant_(i, const_value)
            #     nn.init.xavier_uniform_(i)
            #     nn.init.kaiming_normal_(i)
            # torch.nn.init.uniform_(i)
            # torch.nn.init.normal_(i, 0, std)

        # for i in self.k_adapter_distill:
        #     # nn.init.zeros_(i)
        #     # nn.init.normal_(i, 0,std=1/self.d_k)
        #     if i != None:
        #         # nn.init.zeros_(i)
        #         # nn.init.trunc_normal_(i)
        #         nn.init.normal_(i, 0,1/(self.d_k))
        # for i in self.v_adapter_distill:
        #     # nn.init.zeros_(i)
        #     # nn.init.normal_(i, 0,std=1/self.d_k)
        #     if i != None:
        #         # nn.init.zeros_(i)
        #         # nn.init.trunc_normal_(i)
        #         nn.init.normal_(i, 0,1/(self.d_k))
        #
        # for i in self.k_adapter_img_distill:
        #     # nn.init.zeros_(i)
        #     # nn.init.normal_(i, 0,std=1/self.d_k)
        #     if i != None:
        #         # nn.init.zeros_(i)
        #         # nn.init.trunc_normal_(i)
        #         nn.init.normal_(i, 0,1/(self.d_k))
        #
        # for i in self.v_adapter_img_distill:
        #     # nn.init.zeros_(i)
        #     # nn.init.normal_(i, 0,std=1/self.d_k)
        #     if i != None:
        #         # nn.init.zeros_(i)
        #         # nn.init.trunc_normal_(i)
        #         nn.init.normal_(i, 0,1/(self.d_k))

        self.k_adapter = self.k_adapter.half()
        self.v_adapter = self.v_adapter.half()
        self.k_adapter_img = self.k_adapter_img.half()
        self.v_adapter_img = self.v_adapter_img.half()

        # self.k_adapter_distill = self.k_adapter_distill.half()
        # self.v_adapter_distill = self.v_adapter_distill.half()
        # self.k_adapter_img_distill = self.k_adapter_img_distill.half()
        # self.v_adapter_img_distill = self.v_adapter_img_distill.half()
        # self.linear_adapter = self.linear_adapter.half()

    def forward(self, image, label=None,meta = False):
        # self.prompt_learner.training = False
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # test-time
        # if self.prompt_learner.training is False:
        #     batch = image.shape[0]
        #     views = image.shape[1]
        #     image = image.reshape(image.shape[0] * image.shape[1],image.shape[2],image.shape[3],image.shape[4])


        k_adapters = []
        v_adapters = []
        img_k_adaptrers = []
        img_v_adapters = []


        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()


        # shared_ctx = self.swiglu(shared_ctx)
        shared_ctx = self.dropout_prompt(shared_ctx)
        # for i in self.k_adapter:
        #     k_adapters.append(self.text_mlp_adapter(i))
        # for i in self.v_adapter:
        #     v_adapters.append(self.text_mlp_adapter(i))
        # for i in self.k_adapter_img:
        #     img_k_adaptrers.append(self.image_mlp_adapter(i))
        # for i in self.v_adapter_img:
        #     img_v_adapters.append(self.image_mlp_adapter(i))
        # text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text, k_adapter,
        #                                   v_adapter)
        # image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision,
        #                                     k_adapter_img, v_adapter_img)
        # if self.cfg.DATASET.SUBSAMPLE_CLASSES == "new":
        #     isnew = True
        # else:
        #     isnew = False
        # origin_image_features = self.clip_mode.encode_image(image.type(self.dtype))
        # similarity_scores = F.cosine_similarity(origin_image_features.unsqueeze(1), self.key_img.unsqueeze(0), dim=-1) # [batch,1,dim] * [1,nums,dim]
        # # 根据相似度选择最相似的 k_adapter 和 v_adapter
        # if self.prompt_learner.training:
        #     _, top_k_indices = torch.topk(similarity_scores, k=3, dim=-1)
        #     # 根据相似度选出的 k_adapter 和 v_adapter
        #     # top_k_indices = top_k_indices.squeeze(1)
        #     random_choice = torch.randint(0, 3, (top_k_indices.size(0),)).to(top_k_indices.device) # size(0) = batch size
        #     # 根据随机选择的索引，从 top_k_indices 中选出最终的索引
        #     top_k_indices = top_k_indices.gather(-1, random_choice.unsqueeze(-1)).squeeze(-1)
        #     print(top_k_indices)
        # else:
        #     _, top_k_indices = torch.topk(similarity_scores, k=1, dim=-1)
        #     top_k_indices = top_k_indices.squeeze(1)
        #     print(top_k_indices)
        # selected_k_adapter = torch.stack([self.k_adapter[i] for i in top_k_indices.squeeze(1)], dim=0) # [22, 11, 4, 512]
        # selected_v_adapter = torch.stack([self.v_adapter[i] for i in top_k_indices.squeeze(1)], dim=0)

        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, self.k_adapter_img, self.v_adapter_img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_encoder(prompts, tokenized_prompts,deep_compound_prompts_text,self.k_adapter,self.v_adapter)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        # for i in range(self.prompt_nums):
        #     text_features,text_memory,text_memory_distill = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text,self.k_adapter[i],self.v_adapter[i],None,None,isnew)
        #     all_text_features.append(text_features)
        # for i in range(image_features.shape[0]):
        #     mImg_feature = image_features[i].unsqueeze(0)
        #     # mImg_feature = mImg_feature / mImg_feature.norm(dim=-1, keepdim=True)
        #     mtext_feature = all_text_features[top_k_indices[i]]
        #     mtext_feature = mtext_feature / mtext_feature.norm(dim=-1, keepdim=True)
        #     logit = logit_scale * mImg_feature @ mtext_feature.t()
        #     logits.append(logit)

        # text_features = torch.stack(all_text_features,dim=0)
        # logits = torch.stack(logits, dim=0).squeeze(1)
        # text_features, text_memory, text_memory_distill = self.text_encoder(prompts, tokenized_prompts,
        #                                                                     deep_compound_prompts_text,
        #                                                                     selected_k_adapter, selected_v_adapter,
        #                                                                     None, None, isnew)
        # text_memory = torch.stack(text_memory,dim=1)
        # text_memory_distill = torch.stack(text_memory_distill,dim=1)
        #
        # image_memory = torch.stack(image_memory, dim=1)
        # image_memory_distill = torch.stack(image_memory_distill, dim=1)
        # print(image_memory.shape)
        # print(image_memory_distill.shape)
        # print(text_memory.shape)
        # print(text_memory_distill.shape)
        # exit(0)
        # distill_loss = F.mse_loss(text_memory.detach(),text_memory_distill) + F.mse_loss(image_memory.detach(),image_memory_distill)
        # print(distill_loss)
        # select_features = self.key_img[top_k_indices]

        if self.cfg.EVAL is False:
            photo_features = self.ctx_photo.to(text_features.device)
        if self.prompt_learner.training:
            # print(text_features.shape)
            # print(photo_features.shape)
            # exit(0

            m2_loss = 7.5 * F.mse_loss(text_features, photo_features)

            # sim_loss = 7.5 * F.mse_loss(select_features,image_features)
            # sim_loss = 0.5 * F.cross_entropy(similarity_scores,top_k_indices)
            # m2_loss = 7.5 *  F.mse_loss(text_features, photo_features)

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # logits = logit_scale * image_features @ text_features.t() #余弦相似度



        # test-time
        # if self.prompt_learner.training is False: # [batch,n_views,label]#[batch,n_views,3,224,224]
        #     logits = logits.reshape(batch,views,-1)
        #     logits_new,_ = self.select_confident_samples(logits,0)
        #     weights = torch.linspace(1, 0, logits_new.shape[1]).view(1, logits_new.shape[1], 1).to(logits_new.device)
        #     # print("weights:")
        #     # print(weights)
        #     # exit(0)
        #     logits_new_copy = copy.deepcopy(logits_new)
        #     logits_new = torch.mean(logits_new,dim=1,keepdim=False)
        #     return logits_new,logits_new_copy



        # logits = self.adjust_cos(image_features,text_features)
        if self.prompt_learner.training:
            loss = F.cross_entropy(logits, label)  #+ F.cross_entropy(logits_photo,label)

            return  loss + m2_loss #+  7.5 * F.mse_loss(image_features,label_text_features) #+ self.class_loss(text_features,image_features,label,isText= False)

        return logits

    # def parameters2tensor(self,params):
    #     tensors = None
    #     for i in params:
    #         if tensors is None:
    #             tensors = i
    #         else:
    #             tensors = torch.cat([tensors,i],dim=0)
    #     return tensors
    # def tensor2parameters(self,tensors):
    #     lists = [tensors[i,:,:].unsqueeze(0) for i in range(tensors.shape[0])]
    #     return lists

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def train(self):

        super().train()


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        adapter = "adapter"
        ln = "ln"
        # a = self.model.named_parameters()
        for name, param in self.model.named_parameters():
            # print(name)
            # if "prompt_learner.ctx" in name:
            #     param.requires_grad_(False)
            #     continue
            # print(name)
            if "clip_model" in name:
                param.requires_grad(False);
            if "compound_prompts_text" in name:
                param.requires_grad_(False)
                continue
            if "compound_prompt_projections" in name:
                param.requires_grad_(False)
                continue
            if "ctx_photo" in name:
                param.requires_grad_(False)
                continue
            if adapter in name or name_to_update in name or ("add_" in name) or "key_img" in name:#or (ln in name and "image_encoder" in name) :
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

            # if name_to_update not in name:
            #     # Make sure that VPT prompts are updated
            #     if "VPT" in name:
            #         param.requires_grad_(True)
            #     else:
            #         param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        if cfg.EVAL is False:
            if cfg.DATASET.SUBSAMPLE_CLASSES == "base":
                self.model.ctx_photo = torch.load("photo/"+cfg.DATASET.NAME+".pth").half()
            elif cfg.DATASET.SUBSAMPLE_CLASSES == "new":
                self.model.ctx_photo = torch.load("photo/"+cfg.DATASET.NAME+"_test.pth").half()
                self.model.processNew()
            elif cfg.DATASET.SUBSAMPLE_CLASSES == "all":
                self.model.ctx_photo = torch.load("photo/" + cfg.DATASET.NAME + "_all.pth").half()
            else:
                raise "I don't know the subsample class!"
        # self.model.ctx_photo = torch.load("photo/" + cfg.DATASET.NAME + "_all.pth").half()
            self.model.ctx_photo.requires_grad_(False)
        self.model.to(self.device)








        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # self.optim_meta = build_optimizer(self.model,  cfg.OPTIM)
        # # self.sched_meta = build_lr_scheduler(self.optim_meta, cfg_etmp)
        # self.sched_meta  = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optim_meta, float(35))
        # cfg2 = cfg.OPTIM
        # cfg2.MAX_EPOCH = 20
        # self.sched_meta =  build_lr_scheduler(self.optim, cfg2)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch,meta = False):
        # if not meta:
        image, label = self.parse_batch_train(batch)
        # torch.autograd.set_detect_anomaly(True)
        # torch.autograd.detect_anomaly(True)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:

            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()


        return loss_summary
        # else:
        #     image, label = self.parse_batch_train(batch)
        #     # torch.autograd.set_detect_anomaly(True)
        #     # torch.autograd.detect_anomaly(True)
        #     model = self.model
        #     optim = self.optim_meta
        #     scaler = self.scaler
        #
        #     prec = self.cfg.TRAINER.MAPLE.PREC
        #     if prec == "amp":
        #         with autocast():
        #             loss = model(image, label,meta)
        #         optim.zero_grad()
        #         scaler.scale(loss).backward()
        #         scaler.step(optim)
        #         scaler.update()
        #     else:
        #
        #         loss = model(image, label,meta)
        #         optim.zero_grad()
        #         loss.backward()
        #         optim.step()
        #
        #     loss_summary = {"loss": loss.item()}
        #
        #     if (self.batch_idx + 1) == self.num_batches:
        #         self.sched_meta.step()
        #
        #     return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
