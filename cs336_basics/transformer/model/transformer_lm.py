import math

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
import torch

from cs336_basics.transformer.model import Linear, softmax, RotaryPositionalEmbedding


