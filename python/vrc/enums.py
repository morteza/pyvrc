from enum import Enum


class DecoderType(Enum):
  """Defines which decoding algorithm to use."""

  ONE_HOT = 'one-hot'
  SNR = 'snr'
