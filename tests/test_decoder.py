import pytest

import numpy as np
import scipy
import scipy.stats as stats

import seaborn as sns

import vrc


@pytest.mark.parametrize('params_fixture', ['noisy_params', 'noiseless_params'])
def test_decoding(symbols, stimulus, params_fixture, plt, request):

  params = request.getfixturevalue(params_fixture)

  signal_freq = params['signal_freq']
  timeout_in_sec = params['timeout_in_sec']

  encode = vrc.OneHotEncoder(symbols, signal_freq=signal_freq, noise_freq=0)
  spike_trains = encode(stimulus, timeout_in_sec)

  decode = vrc.OneHotDecoder(symbols, signal_freq=signal_freq, noise_freq=0)
  posteriors = decode(spike_trains, timeout_in_sec)

  # last posterior of the decoded stimulus must be pretty strong
  stimulus_idx = symbols.index(stimulus)
  response_idx = np.argmax(posteriors[:, -1])
  assert stimulus_idx == response_idx

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  # plot posteriors
  sns.lineplot(data=posteriors.T, marker='o', ax=axes[0])
  axes[0].set_xlabel('Time (timepoint)')
  axes[0].set_ylabel('Posterior (p)')

  fig.tight_layout()
  fig.suptitle(f'OneHot Decoder (stimulus = {stimulus_idx})')

  # plot entropies
  entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)
  sns.lineplot(data=entropies.T, marker='o', ax=axes[1])
  axes[1].set_xlabel('Time (timepoint)')
  axes[1].set_ylabel('Entropy (bit)')


def test_decoding_logic(plt):
  signal_freq = 1.5
  # spike_trains = np.array([[1],
  #                          [2, 3],
  #                          [1, 1.1, 1.5, 2.1, 2.7, 3.1, 4.1]])

  spike_counts = np.array([[1, 1, 1, 1],
                           [0, 1, 2, 2],
                           [1, 3, 5, 7]])

  priors = np.zeros_like(spike_counts, dtype='float')

  priors[:, 0] = [.3, .3, .3]

  timestamps = np.linspace(0, spike_counts.shape[1], spike_counts.shape[1] + 1)

  # Poisson Process: rate = events/time * time
  expected_rates = timestamps[1:] * signal_freq

  likelihoods = stats.poisson.pmf(spike_counts, expected_rates)

  likelihoods = scipy.special.softmax(likelihoods, axis=0)

  # Note: prior_t uses prior_{t-1}, so cannot use vectorize or apply_along_axis
  for t in range(1, priors.shape[1]):
    priors[:, t] = priors[:, t - 1] * likelihoods[:, t]

  # normalize (each column sums up to 1)
  priors = priors / np.sum(priors, axis=0)

  assert np.all(priors > 0)

  # stimulus (index=2) must have the highest probebility during the last inference
  assert np.max(priors[:, -1]) == priors[2, -1]

  # plot posteriors
  sns.lineplot(data=priors.T, marker='o')
  plt.xlabel('Time (timepoint)')
  plt.ylabel('Posterior (p)')
  plt.suptitle('Posteriors over time (stimulus = 2)')


def test_scipy_poisson():
  counts = [[1, 1, 1, 1],
            [0, 1, 2, 2],
            [1, 3, 5, 7]]

  rates = [1.5, 3., 4.5, 6.]

  likelihoods = stats.poisson.pmf(counts, rates)
  scipy.special.softmax(likelihoods, axis=0)


def test_exp_normalization():

  def exp_normalize(x):
      b = x.max()
      y = np.exp(x - b)
      return y / y.sum()

  exp_normalize(np.array([.01, .02, .05]))
