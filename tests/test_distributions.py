import numpy as np
import seaborn as sns
import random

import vrc


def test_plot_distributions(plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freqs = [75]
  noise_freqs = [75 / 66]
  decision_entropies = [.7]
  inference_freq = 100
  timeout_in_sec = 10

  # 1. generate some response times using ground truth parameters

  stimuli = random.choices(symbols, k=100)

  all_rts = []
  legends = []
  for s, n, h in zip(signal_freqs, noise_freqs, decision_entropies):
    transmit = vrc.Transmitter(symbols,
                               s,
                               n,
                               inference_freq,
                               h,
                               timeout_in_sec)
    vtransmit = np.vectorize(transmit)
    responses, response_times = vtransmit(stimuli)
    corrects = (responses == stimuli)
    accuracy = int(corrects.mean() * 100)
    all_rts.append(response_times)
    legends.append(f'S/N/H={s}/{n}/{h} (%correct={accuracy:.1f})')

  _, ax = plt.subplots(1, 1)
  sns.set()
  sns.histplot(all_rts, kde=True, label="response time (s)", element='step')
  ax.set(xlabel='Response time (s)')
  ax.legend(legends)
