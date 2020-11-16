# %%
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as pyplot

import vrc


def test_noisy_transmission(symbols, message, noiseless_params):

  noise_freq = noiseless_params['signal_freq'] / 2

  transmit = vrc.Transmitter(symbols,
                             noiseless_params['signal_freq'],
                             noise_freq,
                             noiseless_params['inference_freq'],
                             noiseless_params['decision_entropy'],
                             noiseless_params['timeout_in_sec'])

  pred_message, _ = transmit(message)

  # for reasonable signal/noise freqs, the message must be successfully passed.
  assert pred_message is message


def test_entropy_trace(symbols, plt: pyplot):

  signal_freq = 5
  noise_freq = 10
  inference_freq = 10
  timeout_in_sec = 5
  entropy_threshold = 0.4

  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)
  decode = vrc.SNRDecoder(symbols, signal_freq, noise_freq, inference_freq)

  message = random.choice(symbols)

  spike_trains = encode(message, timeout_in_sec)
  posteriors = decode(spike_trains, timeout_in_sec)
  entropies = - np.sum(posteriors * np.log2(posteriors), axis=0)

  print('posterior_at_stop_time: ', posteriors[:, -1])

  fig, ax = plt.subplots(1, 1)

  sns.set()
  xx = np.linspace(0, timeout_in_sec, entropies.shape[0])
  sns.lineplot(x=xx,
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
