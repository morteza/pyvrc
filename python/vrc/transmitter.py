from dataclasses import dataclass
import numpy as np

import vrc


@dataclass
class Transmitter():
  """Variable Rate Coding - Multi-channel noisy transmitter

  The transmiter returns the decoded response and respective response time
  (in secend). Decoded response represents the trasmitted stimulus to the
  receiver, and response time indicates when the posterior entropy reached
  the threshold on the receiver's side. If the decision is inconclusive during
  the time before timeout, an empty response will be produced and transmission
  time will equal to the timeout.
  Transmitter also produces a trace that shows posterior and posterior entropy
  over time.


  Note: initial entropy must indicate a weak belief with equal probability for
        each symbol (weakest entropy is log(N) for N channels).

  Note: Unit of all timings is seconds, and all frequencies are in Hz (1/sec).

  Attributes:
  -----------
    stimulus (list):
      symbols to encode and transmit. Currently only one symbol can be
      transmitted.
    symbols (list):
      list of all possible symbols.
    signal_freq (float):
      rate of the poisson process that passes the signal symbol (Hz).
    noise_freq (float):
      rate of the poisson process that passes non-signal symbols (Hz).
    timeout_in_sec (float):
      maximum communication duration (seconds).
    sampling_rate (int):
      number of times that decoder samples the spikes (Hz).
    decision_threshold (float):
      entropy decision threshold (bits).

  Example:
  --------
    >>> transmit = Transmitter(symbols = list('ABCD'), 10, 2, 10, .5, 10)
    >>> respons, response_time = transmit('A')


  """
  symbols: list
  signal_freq: float
  noise_freq: float
  inference_freq: float
  decision_threshold: float
  timeout_in_sec: float
  decoder_type: 'vrc.DecoderType' = 'snr'

  # see `vrc.DecoderType` for valid values.

  def __post_init__(self) -> None:

    self.encode = vrc.OneHotEncoder(self.symbols,
                                    self.signal_freq,
                                    self.noise_freq)

    if self.decoder_type == vrc.DecoderType.SNR.value:
      assert self.noise_freq > 0, 'SNR decoder requires noise_freq>0'
      snr = (self.signal_freq + self.noise_freq) / self.noise_freq
      self.decode = vrc.SNRDecoder(self.symbols,
                                   snr,
                                   self.inference_freq)
    else:
      print('Warning: one-hot decoder is not stable!')
      self.decode = vrc.OneHotDecoder(self.symbols,
                                      self.signal_freq,
                                      self.noise_freq,
                                      self.inference_freq)

  def __call__(self, stimulus):
    """Encodes a stimulus, transmits it through multiple noisy channels, and decodes it.

    Returns:
    --------
      A tuple of predicted stimulus and transmission time in that order.
      The type of the output is as follows: (char, float).

    """

    # Encode
    spike_trains = self.encode(stimulus, self.timeout_in_sec)

    # Decode
    posteriors = self.decode(spike_trains, self.timeout_in_sec)
    entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)

    decision_timepoint = np.argmax(entropies < self.decision_threshold)

    if decision_timepoint > 0:
      response_index = np.argmax(
          posteriors[:, decision_timepoint]
      ).astype(int)
      response = self.symbols[response_index]
      # TODO: use timestamps instead of the following
      response_time = decision_timepoint / self.inference_freq
    else:
      # decision was inconclusive
      response = None
      response_time = None

    return response, response_time

  def add_noise(self):
    raise NotImplementedError('Use `noise_freq` parameter instead.')
