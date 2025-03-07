import pytest
import numpy as np

import seaborn as sns

import vrc


@pytest.mark.parametrize('params_fixture', ['noisy_params', 'noiseless_params'])
def test_noisy_decoding(symbols, stimulus, params_fixture, plt, request):

  params = request.getfixturevalue(params_fixture)

  signal_freq = params['signal_freq']
  noise_freq = params['noise_freq']
  timeout_in_sec = params['timeout_in_sec']

  # to avoid division-by-zero in SNR
  if params['noise_freq'] == 0:
    noise_freq = signal_freq / 2.0

  snr = signal_freq / noise_freq

  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)
  spike_trains = encode(stimulus, timeout_in_sec)

  decode = vrc.SNRDecoder(symbols, snr, noise_freq=noise_freq)

  posteriors = decode(spike_trains, timeout_in_sec)

  # strongest posterior identifies the response
  stimulus_idx = symbols.index(stimulus)
  response_idx = np.argmax(posteriors[:, -1])
  assert stimulus_idx == response_idx

  fig, axes = plt.subplots(1, 2, figsize=(10, 5))

  # plot posteriors
  sns.lineplot(data=posteriors.T, marker='o', ax=axes[0])
  axes[0].set_xlabel('Time (s)')
  axes[0].set_ylabel('Posterior (p)')

  fig.tight_layout()
  fig.suptitle(f'SNR Decoder (stimulus = {stimulus_idx})')

  # plot entropies
  entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)
  sns.lineplot(data=entropies.T, marker='o', ax=axes[1])
  axes[1].set_xlabel('Time (timepoint)')
  axes[1].set_ylabel('Entropy (bit)')


def test_snr_decoding_logic(plt):
  # signal_freq = 1.5, noise_freq=0.5
  snr = 1.1
  # spike_trains = np.array([[1],
  #                          [2, 3],
  #                          [1, 1.1, 1.5, 2.1, 2.7, 3.1, 4.1]])

  spike_counts = np.array([[1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 1, 2, 2, 3],
                           [0, 0, 0, 1, 2, 2, 3],
                           [1, 1, 1, 3, 5, 5, 7]])

  priors = np.array([[.25, .25, .25, .25]])

  posteriors = priors.T * (snr ** spike_counts)
  posteriors = posteriors / np.sum(posteriors, axis=0)

  entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)

  # plot posteriors
  sns.lineplot(data=entropies.T, marker='o')
  plt.xlabel('Time (timepoint)')
  plt.ylabel('Remaining entropy (bits)')
  plt.suptitle('Entropy over time (stimulus = 3)')
