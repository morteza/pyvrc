import numpy as np

import vrc


def test_uninformed_priors():
  symbols = list('ABCD')

  priors = [1 / len(symbols) for _ in symbols]
  assert priors == [.25, .25, .25, .25]

  entropy = - np.sum(priors * np.log2(priors), axis=0)
  assert entropy == 2.0


def test_initial_prior_in_snr_decoder():
  symbols = list('ABCD')
  stimulus = symbols[1]
  signal_freq = 10
  noise_freq = 10
  timeout = 10  # seconds

  uninformed_priors = [1 / len(symbols) for _ in symbols]

  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)
  decode = vrc.SNRDecoder(symbols, signal_freq, noise_freq)

  spike_trains = encode(stimulus, timeout)
  # print([s[0] for s in spike_trains.values()])
  # initial_priors=None
  posteriors = decode(spike_trains, timeout)

  assert np.all(np.isclose(posteriors[:, 0], uninformed_priors))
