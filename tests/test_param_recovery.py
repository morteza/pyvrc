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
    legends.append(f'S/N/H={s}/{n}/{h} (%correct={accuracy:.1f})')

  _, ax = plt.subplots(1, 1)
  sns.set()
  sns.histplot(rts, kde=True, label="RT (s)", element='step')
  ax.set(xlabel='Response time (s)')
  ax.legend(legends)


def test_params_recovery(symbols, plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freq = 40
  noise_freq = 10
  decision_entropy = 0.5
  inference_freq = 100
  timeout_in_sec = 10

  n_stimuli = 100
  n_simulations = 100
  n_ax_trials = 10

  # 1. generate/simulate response times using ground truth model parameters

  transmit = vrc.Transmitter(symbols,
                             signal_freq,
                             noise_freq,
                             inference_freq,
                             decision_entropy,
                             timeout_in_sec)

  data = pd.DataFrame({
      'stimulus': np.array(random.choices(symbols, k=n_stimuli))
  })

  data[['response', 'rt']] = data['stimulus'].apply(transmit).apply(pd.Series)

  true_accuracy = data.query('stimulus == response')['rt'].count() / len(data)

  valids = ~data['rt'].isna()

  print(f'Ground truth: {true_accuracy*100:.2f}% correct in {len(data)} trials.')

  # 2. now define and fit a model to the groupnd truth RTs
  model = vrc.BayesPoissonModel(symbols,
                                timeout_in_sec,
                                inference_freq=inference_freq,
                                backend='ax',
                                simulations_count=n_simulations,
                                ax_total_trials=n_ax_trials)

  recovered_params = model.fit(data['rt'], data['stimulus'])
  print('FITTED PARAMS:', recovered_params)

  # 3. next is to use fitted parameters and simulate again
  transmit = vrc.Transmitter(symbols,
                             recovered_params['signal_freq'],
                             recovered_params['noise_freq'],
                             inference_freq,
                             recovered_params['decision_entropy'],
                             timeout_in_sec)

  data[['recovered_response', 'recovered_rt']] = \
      data['stimulus'].apply(transmit).apply(pd.Series)

  recovered_accuracy = data.query('stimulus == recovered_response')['rt'].count() / len(data)

  print('medians:', data['rt'].median(), data['recovered_rt'].median())

  data = data.melt(value_vars=['rt', 'recovered_rt'], var_name='kind', value_name='rt')

  # plot ground truth RTs vs recovered RTs
  _, ax = plt.subplots(1, 1)
  binwidth = 1. / inference_freq
  sns.histplot(data,
               x='rt', hue='kind',
               binwidth=binwidth, kde=True, element='step', label="RT (s)")
  ax.set(xlabel='Response time (s)')

  legends = [f'Truth (%correct={true_accuracy * 100:.1f}%)',
             f'Recovered (%correct={recovered_accuracy * 100:.1f}%)']
  ax.legend(legends)

  plt.suptitle('Ground Truth v.s. Recovered distributions')

  # 4. compare recovered parameters to the ground truth
  # assert np.isclose(accuracies.mean(), recovered_accuracies.mean())
  # assert np.isclose(signal_freq, best_params['signal_freq'])
  # assert np.isclose(noise_freq, best_params['noise_freq'])
  # assert np.isclose(decision_entropy, best_params['decision_entropy'])
