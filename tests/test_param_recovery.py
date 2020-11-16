import random

import numpy as np

import vrc

import seaborn as sns


def test_non_identifiable_params(plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freqs = [19.196080653727883, 54.56850742921233]
  noise_freqs = [16.61880410112117, 70.27448069564998]
  decision_entropies = [1.5515815711871377, 1.1339395301416517]
  inference_freq = 100
  timeout_in_sec = 10

  # 1. generate/simulate response times using ground truth model parameters

  sent_msgs = random.choices(symbols, k=1000)

  for i in range(2):
    transmit = vrc.Transmitter(symbols,
                               signal_freqs[i],
                               noise_freqs[i],
                               inference_freq,
                               decision_entropies[i],
                               timeout_in_sec)
    vtransmit = np.vectorize(transmit)
    pred_msgs, rts = vtransmit(sent_msgs)
    accuracies = (pred_msgs == sent_msgs)
    sns.distplot(rts, axlabel="RT (s)")
    print(np.nan_to_num(accuracies).mean())


def test_two_dists():
  """simulate two distributions using same hyperparameters."""
  pass


def test_params_recovery(symbols, plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freq = 100.0
  noise_freq = 10.0
  decision_entropy = 0.4
  inference_freq = 1000
  timeout_in_sec = 10

  # 1. generate/simulate response times using ground truth model parameters

  transmit = vrc.Transmitter(symbols,
                             signal_freq,
                             noise_freq,
                             inference_freq,
                             decision_entropy,
                             timeout_in_sec)

  sent_msgs = random.choices(symbols, k=100)

  vtransmit = np.vectorize(transmit)
  pred_msgs, rts = vtransmit(sent_msgs)
  accuracies = (pred_msgs == sent_msgs)

  print('Accuracy:',
        accuracies.mean(),
        'in',
        len(sent_msgs),
        'simulated trials')

  # 2. now fit a model to the generated RTs
  model = vrc.BayesPoissonModel(symbols,
                                timeout_in_sec,
                                inference_freq=100,
                                backend='ax',
                                ax_total_trials=5)

  best_params = model.fit(rts, sent_msgs)
  print('BEST MODEL PARAMS :', best_params)

  transmit = vrc.Transmitter(symbols,
                             best_params['signal_freq'],
                             best_params['noise_freq'],
                             inference_freq,
                             best_params['decision_entropy'],
                             timeout_in_sec,
                             decoder_type='snr')

  vtransmit = np.vectorize(transmit)
  pred_msgs, recovered_rts = vtransmit(sent_msgs)
  recovered_accuracies = (pred_msgs == sent_msgs)

  # plot ground truth RTs vs recovered RTs
  sns.distplot(rts, axlabel='Ground Truth')
  sns.distplot(recovered_rts, axlabel='Recovered')
  plt.suptitle(f'Ground Truth Accuracy: {accuracies.mean()} \n'
               f'Recovered Accuracy: {recovered_accuracies.mean()}')

  # 3. compare recovered parameters to the ground truth
  # assert np.isclose(accuracies.mean(), recovered_accuracies.mean())
  # assert np.isclose(signal_freq, best_params['signal_freq'])
  # assert np.isclose(noise_freq, best_params['noise_freq'])
  # assert np.isclose(decision_entropy, best_params['decision_entropy'])

  # TODO simulate new RTs and plot them against ground truth
