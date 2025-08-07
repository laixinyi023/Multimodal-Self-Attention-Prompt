"""
Trainers module for MemoryUnit project.

This module contains various trainer implementations for different prompt learning methods
including CoOp, Co-CoOp, MaPLe, VPT, and other CLIP-based approaches.
"""

from .coop import CoOp
from .cocoop import CoCoOp
from .maple import MaPLe
from .vpt import VPT
from .independentVL import IndependentVL
from .zsclip import ZeroShotCLIP

__all__ = [
    'CoOp',
    'CoCoOp', 
    'MaPLe',
    'VPT',
    'IndependentVL',
    'ZeroShotCLIP'
]
