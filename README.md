# MSP: Multimodal Self-attention Prompt learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of **MSP: Multimodal Self-attention Prompt learning**, a novel approach for few-shot learning with vision-language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

MSP (Multimodal Self-attention Prompt learning) is a novel prompt learning method that leverages self-attention mechanisms to learn effective prompts for both vision and language modalities in few-shot learning scenarios. The method extends traditional prompt learning approaches by incorporating cross-modal attention and adaptive prompt generation.

### Key Contributions

- **Multimodal Prompt Learning**: Learns prompts for both vision and language modalities simultaneously
- **Self-attention Mechanism**: Utilizes attention mechanisms to adapt prompts dynamically
- **Cross-modal Alignment**: Ensures alignment between visual and textual representations
- **Few-shot Adaptation**: Effective adaptation to new classes with limited samples

## ✨ Features

- **Multiple Prompt Learning Methods**: Implementation of CoOp, Co-CoOp, MaPLe, VPT, and MSP
- **Comprehensive Dataset Support**: Support for 11 standard few-shot learning datasets
- **Flexible Architecture**: Modular design for easy extension and customization
- **Reproducible Results**: Complete implementation with detailed configuration files
- **Memory Efficient**: Optimized for training with limited computational resources

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.7+
- CUDA (for GPU training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/laixinyi023/Multimodal-Self-Attention-Prompt.git
   cd MemoryUnit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv msp_env
   source msp_env/bin/activate  # On Windows: msp_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

### Dataset Preparation

Download and prepare the datasets according to the [DATASETS.md](docs/DATASETS.md) guide.

## 🏃‍♂️ Quick Start

### Training MSP on ImageNet

```bash
# Train MSP on ImageNet with 16-shot learning
python tools/train.py \
    --root /path/to/datasets \
    --trainer MSP \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/MSP/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --output-dir output/msp_imagenet \
    --seed 1
```

### Evaluation

```bash
# Evaluate the trained model
python tools/train.py \
    --root /path/to/datasets \
    --trainer MSP \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/MSP/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --model-dir output/msp_imagenet \
    --load-epoch 50 \
    --eval-only
```

## 📁 Project Structure

```
MemoryUnit/
├── clip/                    # CLIP model implementation
│   ├── __init__.py
│   ├── clip.py             # Core CLIP functionality
│   ├── model.py            # CLIP model architecture
│   ├── attention.py        # Attention mechanisms
│   └── Adapter.py          # Adapter modules
├── configs/                 # Configuration files
│   ├── datasets/           # Dataset configurations
│   └── trainers/           # Trainer configurations
├── datasets/               # Dataset implementations
│   ├── __init__.py
│   ├── imagenet.py         # ImageNet dataset
│   ├── caltech101.py       # Caltech-101 dataset
│   └── ...                 # Other datasets
├── trainers/               # Trainer implementations
│   ├── __init__.py
│   ├── maple.py           # MaPLe trainer
│   ├── coop.py            # CoOp trainer
│   ├── cocoop.py          # Co-CoOp trainer
│   ├── vpt.py             # VPT trainer
│   └── independentVL.py   # Independent VL trainer
├── scripts/               # Training and evaluation scripts
│   ├── maple/             # MaPLe scripts
│   ├── coop/              # CoOp scripts
│   └── ...                # Other method scripts
├── docs/                  # Documentation
├── tools/                 # Utility tools
└── README.md             # This file
```

## 📖 Usage

### Configuration

The project uses YAML configuration files for easy customization:

- **Dataset Configs**: `configs/datasets/` - Dataset-specific configurations
- **Trainer Configs**: `configs/trainers/` - Method-specific training configurations

### Training Different Methods

#### MaPLe (Multi-modal Adaptive Prompt Learning)
```bash
python tools/train.py \
    --root /path/to/datasets \
    --trainer MaPLe \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --output-dir output/maple_imagenet
```

#### CoOp (Context Optimization)
```bash
python tools/train.py \
    --root /path/to/datasets \
    --trainer CoOp \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/coop_imagenet
```

#### VPT (Visual Prompt Tuning)
```bash
python tools/train.py \
    --root /path/to/datasets \
    --trainer VPT \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --config-file configs/trainers/VPT/vit_b16_c2_ep5_batch4_4.yaml \
    --output-dir output/vpt_imagenet
```

### Cross-Dataset Evaluation

```bash
# Evaluate on ImageNet-A
python tools/train.py \
    --root /path/to/datasets \
    --trainer MSP \
    --dataset-config-file configs/datasets/imagenet_a.yaml \
    --config-file configs/trainers/MSP/vit_b16_c2_ep5_batch4_2ctx.yaml \
    --model-dir output/msp_imagenet \
    --load-epoch 50 \
    --eval-only
```

## 🙏 Acknowledgments

- [CLIP](https://github.com/openai/CLIP) - Original CLIP implementation
- [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) - Domain adaptation framework
- [CoOp](https://github.com/KaiyangZhou/CoOp) - Context optimization baseline
- [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) - Multi-modal prompt learning baseline

