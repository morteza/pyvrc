from enum import Enum


class DecoderType(Enum):
  """scpecifies which decoding algorithm to use."""

  ONE_HOT = 'one-hot'
  SNR = 'snr'


class OptimizerBackend(Enum):
  """Specifies the backend used for hyperparameter optimizations."""

  AX = 'ax'
  # SCIPY = 'scipy'
