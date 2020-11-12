from dataclasses import dataclass

import numpy as np

from .utils import count_spikes


@dataclass
class SNRDecoder(object):
  """Continious stable decoder using signal-to-noise ratio.

  Note: It follows pytorch module convention

  """
  symbols: list
  signal_freq: float
  noise_freq: float
  inference_freq: float = 10

  def __call__(self,
               spike_trains: dict,
               timeout_in_sec: float,
               initial_priors: np.array = None) -> np.array:
    """
    Infer the posterior of receiving a signal meesage in each channel.

    It uses Pytorch-like callable class design pattern.

    Example:
    --------
      >>> d = Decoder(2.)
      >>> d(spike_trains, priors)

    Args:
    -----
    spike_trains (dict):
      overall shape must be (channels * times), and keys represent symbols.
    timeout_in_sec (float):
      inference timeout in seconds. None accuracy and timeout_in_sec response
      time will be generated if entropy does no reach the threshold up to this
      timeout time.
    initial_priors (np.array, optional):
      None or array of `channels` priors

    """
    # convert dict to numpy array and then count spikes
    spike_trains_mat = np.array(list(spike_trains.values()))
    spike_counts = count_spikes(spike_trains_mat,
                                duration=timeout_in_sec,
                                counting_freq=self.inference_freq)

    if initial_priors is None:
      # weak uniform prior
      priors = [1 / len(self.symbols) for _ in self.symbols]
    else:
      priors = initial_priors

    snr = (self.signal_freq + self.noise_freq) / self.noise_freq
    posteriors = np.transpose([priors]) * (snr ** spike_counts)

    # normalize
    posteriors = posteriors / np.sum(posteriors, axis=0)

    return posteriors
