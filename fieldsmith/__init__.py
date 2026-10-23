"""fieldsmith: symbolic classical field theory and gauge theory from LaTeX."""
from .spacetime import Spacetime
from .tensors import Tensor
from .model import Model
from .latex import ParseError

__version__ = "0.1.0"
__all__ = ["Spacetime", "Tensor", "Model", "ParseError"]
