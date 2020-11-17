import pytest

import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import vrc


@pytest.mark.parametrize('params_fixture', ['noisy_params'])
def test_noisy_transmission(symbols, stimulus, params_fixture, request):

  params = request.getfixturevalue(params_fixture)

  transmit = vrc.Transmitter(symbols,
                             params['signal_freq'],
                             params['noise_freq'],
                             params['inference_freq'],
                             params['decision_entropy'],
                             params['timeout_in_sec'])

  response, _ = transmit(stimulus)

  # for reasonable signal/noise freqs, the stimulus must be successfully passed.
  assert response is stimulus


@pytest.mark.parametrize('params_fixture', ['noisy_params', 'noiseless_params'])
def test_confusion_matrix(symbols, params_fixture, plt: pyplot, request):

  params = request.getfixturevalue(params_fixture)

  decoder_type = (vrc.DecoderType.ONE_HOT
                  if params_fixture == 'noiseless_params'
                  else vrc.DecoderType.SNR)

  stimuli = random.choices(symbols, k=1000)

  noise_freq = params['noise_freq']

  if noise_freq == 0:
    # to avoid division-by-zero in SNR
    noise_freq = params['signal_freq'] / 2

  transmit = vrc.Transmitter(symbols,
                             params['signal_freq'],
                             noise_freq,
                             params['inference_freq'],
                             params['decision_entropy'],
                             params['timeout_in_sec'],
                             decoder_type=decoder_type)

  responses, response_times = np.vectorize(transmit)(stimuli)

  responses = ['' if x is None else x for x in responses]

  n_classes = np.unique(responses).shape[0]
  cm = confusion_matrix(stimuli, responses)

  assert cm.shape == (n_classes, n_classes)

  accuracy = cm.trace() / cm.sum()
  print('overall accuracy:', accuracy)

  # plot
  labels = np.unique(symbols + responses)
  _, ax = plt.subplots(1, 1)
  sns.set_style("white")
  ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)


def test_entropy_trace(symbols, plt: pyplot):

  signal_freq = 5
  noise_freq = 10
  inference_freq = 10
  timeout_in_sec = 5
  entropy_threshold = 0.4

  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)
  decode = vrc.SNRDecoder(symbols, signal_freq, noise_freq, inference_freq)

  stimulus = random.choice(symbols)

  spike_trains = encode(stimulus, timeout_in_sec)
  posteriors = decode(spike_trains, timeout_in_sec)
  entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)

  print('posterior_at_stop_time: ', posteriors[:, -1])

  fig, ax = plt.subplots(1, 1)

  xs = np.linspace(0, timeout_in_sec, entropies.shape[0])
  sns.lineplot(x=xs,
               y=entropies,
               drawstyle='steps-post',
               ax=ax)

  decision_point = np.argmax(entropies < entropy_threshold)
  decision_time = decision_point / inference_freq

  fig.suptitle('Entropy')
  ax.axvline(decision_time, color='grey', ls='--', lw=1)
  ax.axhline(entropy_threshold, color='grey', ls='--', lw=1)


# to be executed as cell in jupyter
# test_entropy_trace(list('ABCD'), pyplot)
