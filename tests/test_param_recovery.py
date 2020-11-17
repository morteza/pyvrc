import random
import numpy as np
import pandas as pd
import seaborn as sns

import vrc


def test_params_comparison(plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freqs = [19, 54]
  noise_freqs = [16, 70]
  decision_entropies = [1.0, 1.1]
  inference_freq = 100
  timeout_in_sec = 10

  # 1. generate/simulate response times using ground truth model parameters

  sent_msgs = random.choices(symbols, k=1000)

  rts = []
  legends = []
  for s, n, h in zip(signal_freqs, noise_freqs, decision_entropies):
    transmit = vrc.Transmitter(symbols,
                               s,
                               n,
                               inference_freq,
                               h,
                               timeout_in_sec)
    vtransmit = np.vectorize(transmit)
    pred_msgs, pred_rts = vtransmit(sent_msgs)
    corrects = (pred_msgs == sent_msgs)
    accuracy = int(corrects.mean() * 100)
    rts.append(pred_rts)
    legends.append(f'S/N/H [Accuracy]={s}/{n}/{h} [{accuracy}%]')

  _, ax = plt.subplots(1, 1)
  sns.set()
  sns.histplot(rts, kde=True, label="RT (s)", element='step')
  ax.set(xlabel='Response time (s)')
  ax.legend(legends)


def test_two_dists():
  """simulate two distributions using same hyperparameters."""
  pass


def test_params_recovery(symbols, plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freq = 20
  noise_freq = 20
  decision_entropy = 0.5
  inference_freq = 100
  timeout_in_sec = 10

  # 1. generate/simulate response times using ground truth model parameters

  transmit = vrc.Transmitter(symbols,
                             signal_freq,
                             noise_freq,
                             inference_freq,
                             decision_entropy,
                             timeout_in_sec)

  true_msgs = np.array(random.choices(symbols, k=100))

  vtransmit = np.vectorize(transmit)
  simulated_msgs, true_rts = vtransmit(true_msgs)
  corrects = (simulated_msgs == true_msgs)
  valids = ~pd.isna(true_rts)

  print('Accuracy:',
        corrects.mean(),
        'in',
        len(true_msgs),
        'simulated trials')

  # 2. now fit a model to the generated RTs
  model = vrc.BayesPoissonModel(symbols,
                                timeout_in_sec,
                                inference_freq=inference_freq,
                                backend='ax',
                                simulations_count=100,
                                ax_total_trials=10)

  fitted_params = model.fit(true_rts[valids], true_msgs[valids])
  print('FITTED PARAMS:', fitted_params)

  # 3. next is to use fitted parameters and simulate again
  transmit = vrc.Transmitter(symbols,
                             fitted_params['signal_freq'],
                             fitted_params['noise_freq'],
                             inference_freq,
                             fitted_params['decision_entropy'],
                             timeout_in_sec)

  vtransmit = np.vectorize(transmit)
  pred_msgs, pred_rts = vtransmit(true_msgs)
  pred_corrects = (pred_msgs == true_msgs)

  legends = [f'Truth (accuracy: {corrects.mean() * 100:.2f}%)',
             f'Recovered (accuracy: {pred_corrects.mean() * 100:.2f}%)']

  # plot ground truth RTs vs recovered RTs
  _, ax = plt.subplots(1, 1)
  sns.histplot([true_rts, pred_rts], kde=True, label="RT (s)", element='step')
  ax.set(xlabel='Response time (s)')
  ax.legend(legends)

  plt.suptitle('Ground Truth v.s. Recovered distributions')

  # 4. compare recovered parameters to the ground truth
  # assert np.isclose(accuracies.mean(), recovered_accuracies.mean())
  # assert np.isclose(signal_freq, best_params['signal_freq'])
  # assert np.isclose(noise_freq, best_params['noise_freq'])
  # assert np.isclose(decision_entropy, best_params['decision_entropy'])
