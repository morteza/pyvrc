from dataclasses import dataclass
import numpy as np

from vrc import OneHotDecoder, OneHotEncoder, SNRDecoder


@dataclass
class Transmitter():
  """Variable Rate Coding - Multi-channel noisy transmitter

  The transmiter returns the decoded message and the transmission time
  (in secend). Decoded message represents the trasmitted message to the
  receiver, and transmission time indicates when the posterior entropy reached
  the threshold on the receiver's side. If the decision is inconclusive during
  the time before timeout, an empty message will be produced and transmission
  time will equal to the timeout.
  Transmitter also produces a trace that shows posterior and posterior entropy
  over time.


  Note: initial entropy must indicate a weak belief with equal probability for
        each symbol (weakest entropy is log(N) for N channels).

  Note: Unit of all timings is seconds, and all frequencies are in Hz (1/sec).

  Attributes:
  -----------
    message (list):
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
    entropy_threshold_bits (float):
      entropy decision threshold (bits).

  Example:
  --------
    >>> transmit = Transmitter(symbols = ['A','B'], 10, 2, 1)
    >>> decoded_message, transmission_time, trace = transmit('A')

  Returns:
  --------
    A tuple containing accuracy and transmission time in that order.
    The type of the output is as follows: (float, float).

  """
  symbols: list
  signal_freq: float
  noise_freq: float
  inference_freq: float
  entropy_threshold_bits: float
  timeout_in_sec: float
  decoder_type: str = 'snr'

  def __post_init__(self) -> None:
    self.encode = OneHotEncoder(self.symbols,
                                self.signal_freq,
                                self.noise_freq)

    if self.decoder_type == 'snr':
      assert self.noise_freq > 0, 'SNR decoder requires noise_freq>0'
      snr = (self.signal_freq + self.noise_freq) / self.noise_freq
      self.decode = SNRDecoder(self.symbols,
                               snr,
                               self.inference_freq)
    else:
      self.decode = OneHotDecoder(self.symbols,
                                  self.signal_freq,
                                  self.noise_freq,
                                  self.inference_freq)

  def __call__(self, message):
    """Encodes a message, transmits it through multiple noisy channels, and decodes it.

    """

    # Encode
    spike_trains = self.encode(message, self.timeout_in_sec)

    # Decode
    posteriors = self.decode(spike_trains, self.timeout_in_sec)
    entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)

    decision_index = np.argmax(entropies < self.entropy_threshold_bits)

    if decision_index > 0:
      recvd_msg_index = np.argmax(
          posteriors[:, decision_index]
      ).astype(int)
      recvd_msg = self.symbols[recvd_msg_index]
      accuracy = float(recvd_msg == message)
      # TODO: use timestamps instead of the following
      stop_time = decision_index / self.inference_freq
    else:
      # decision was inconclusive
      accuracy = None
      stop_time = self.timeout_in_sec

    return accuracy, stop_time

  def add_noise(self):
    raise NotImplementedError('Use `noise_freq` parameter instead.')
