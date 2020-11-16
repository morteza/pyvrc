from dataclasses import dataclass

import numpy as np

import scipy
import scipy.stats as stats

from .utils.count_spikes import count_spikes


@dataclass
class OneHotDecoder(object):
  """Variable Rate Coding - One-Hot Decoder

  Note: Following callable object design pattern, you need to first initialize the
  object, and then call it:

  Example:
  --------
  >>> d = Decoder(10, 0)
  >>> d(spike_trains, priors)
  """
  symbols: list
  signal_freq: float
  noise_freq: float = 0
  inference_freq: float = 100

  def __call__(self,
               spike_trains: dict,
               timeout_in_sec: float,
               initial_priors: np.array = None) -> np.array:
    """
    Infers the posterior of receiving meesage from each channel.

    Example:
    --------
      >>> d = Decoder(10, 0)
      >>> d(spike_trains, priors)

    Args:
    -----
    spike_trains (dict):
      overall shape must be (channels * times), and keys represent symbols.
    timeout_in_sec (float):
      inference timeout in seconds. None prediction and response time
      will be generated if entropy does no reach the threshold up to this
      timeout time.
    initial_priors (np.array,optional):
      None or array of a shape (channels * inference_times)

    """

    # convert dict to numpy array and then count spikes
    spike_trains_mat = np.array(list(spike_trains.values()))
    spike_counts = count_spikes(spike_trains_mat,
                                duration=timeout_in_sec,
                                counting_freq=self.inference_freq)

    priors = np.zeros_like(spike_counts)

    if initial_priors is None:
      # use weak uniform prior
      priors[:, 0] = [1 / len(self.symbols) for _ in self.symbols]
    else:
      priors[:, ] = initial_priors

    freq = self.signal_freq + self.noise_freq

    # continously match pmf(freq) to spike_counts/timestamps
    timestamps = np.linspace(0,
                             timeout_in_sec,
                             timeout_in_sec * self.inference_freq + 1)

    # Poisson Process: rate = events/time * time
    expected_rates = timestamps * freq
    likelihoods = stats.poisson.pmf(spike_counts, expected_rates)

    # normalize likelihoods (sums up to 1)
    likelihoods = scipy.special.softmax(likelihoods, axis=0)

    # calculates prior(t) (i.e., posterior), using prior(t-1)
    # Note: prior(t) uses prior(t-1), so cannot use vectorize/apply_along_axis
    for t in range(1, priors.shape[1]):
      priors[:, t] = priors[:, t - 1] * likelihoods[:, t]

    # normalize priors and avoid division-by-zero (each column sums up to 1)
    priors_sum = priors.sum(axis=0)
    priors = np.divide(priors, priors_sum, where=priors_sum > 0)

    return priors
