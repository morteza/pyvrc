from dataclasses import dataclass
import math

import numpy as np
import scipy.stats as stats


@dataclass
class OneHotEncoder():
  """Produces one-hot configurations of mulltiple Poisson spike trains.

  Encoder produces multiple poisson processes which spike with respect to
  expected frequencies. It encodes a stimulus using Poisson rate coding and a
  grain of noise.

  Note that encoding happens on the sender's side. Currently it only support
  One-Hot encoding, i.e., it creates a single neuron per each symbol and that
  neuron only represents a single symbol.


  Args:
  -----
    symbols (list):
      list of all possible symbols.
    signal_freq (float):
      firing rate of stimulus channel (Hz).
    noise_freq(float):
      firing rate of noise channels (Hz).
    homogeneous (boolean, optional):
      generate spikes using a homogeneous poisson process; otherwise generate
      inter-stimulus intervals using exponentional process. Default is True.

  Returns:
  --------
    Callable encoder which can be used like this:
      >>> e = OneHotEncoder(...)
          e(...)

  Raises:
  -------
    ValueError: if symbols list is None or not provided.
    ValueError: signal_rate must be a positive number greater than noise_rate.
  """

  symbols: list
  signal_freq: float
  noise_freq: float = 0
  homogeneous: bool = True

  def __post_init__(self):

    # validate parameters
    assert self.symbols is not None, "Invalid symbols list"

  def __call__(self, stimulus, duration_in_sec: float) -> dict:
    """Encode a stimulus and return spike trains (one train per channel).

    Args:
    -----
      stimulus (str or list):
        Stimuli to be encoded. Must be of size 1 for One-Hot encoding.
        The list format is not supported.
      duration_in_sec (float):
        total duration of spike trains in seconds, which results in producing
        freq*duration spikes.

    Example:
    --------
      The following example produces 3 spike trians of 1 seconds length,
      with signal spikes for 'A', and noise spikes for the rest.
      >>> encode('A', 10)
          {
            'A': array([0.09980255, 0.19373741, 0.24957022, 0.41926107, \
                        0.53529364, 0.60035229, 0.60543173, 0.61491005]),
            'B': array([0.24141811, 0.97706707]),
            'C': array([0.95783754])
          }

    Returns:
    --------
      Channel configurations and spikes, that is a dictionary with channel
      names as keys, and one spike train for each channel; each item represents
      a single poisson process.
      Since only one-hot encoding is supported, channels are named after
      symbols. For example, the following channels configuration will be
      produced for two symbols:
        {
          '<Symbol1>': [1.0, 2.0, 3.0],
          '<Symbol2>': [3., 4.]
        }

    """

    spike_trains = {}

    for s in self.symbols:

      # one-hot configuration (one signal channel + the rest are noise channels)
      freq = self.noise_freq
      if s == stimulus:
        freq += self.signal_freq

      if self.homogeneous:
        # simulate homogeneous poisson process
        spikes_cnt = stats.poisson(freq).rvs(1).astype(float)
        spikes_cnt *= duration_in_sec
        spikes = stats.uniform.rvs(loc=0,
                                   scale=duration_in_sec,
                                   size=spikes_cnt.astype(int))
        spikes = np.sort(spikes)
      else:
        # generate random inter-spike intervals and convert them to spike train
        spikes_cnt = math.ceil(2 * freq * duration_in_sec)
        isi = np.random.exponential(1 / freq, spikes_cnt)
        spikes = np.cumsum(isi)

      # remove out-of-bound spikes
      spikes = spikes[np.where(spikes < duration_in_sec)]

      spike_trains[s] = spikes

    return spike_trains
