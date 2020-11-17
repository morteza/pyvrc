import random
import numpy as np
import pandas as pd
import seaborn as sns
from ax.plot.contour import interact_contour
from ax.plot.render import plot_config_to_html
from ax.utils.report.render import render_report_elements

import vrc


def test_params_recovery(plt):

  # ground truth (TODO: read from test fixture)
  symbols = list('ABCD')
  signal_freq = 20.0
  noise_freq = 10.0
  decision_entropy = .2
  inference_freq = 100
  timeout = 10.0  # in sec

  n_stimuli = 100

  store_ax_report = True

  # 1. generate/simulate response times using ground truth model parameters

  transmit = vrc.Transmitter(symbols,
                             signal_freq,
                             noise_freq,
                             inference_freq,
                             decision_entropy,
                             timeout)

  data = pd.DataFrame({
      'stimulus': np.array(random.choices(symbols, k=n_stimuli))
  })

  data[['response', 'rt']] = data['stimulus'].apply(transmit).apply(pd.Series)

  true_accuracy = data.query('stimulus == response')['rt'].count() / len(data)

  # 2. now fit a model to the groupnd truth RTs

  model = vrc.BayesPoissonModel(symbols,
                                timeout,
                                backend='ax')

  recovered_params, recovered_vals, ax_experiment, ax_model = \
      model.ax_fit(data['rt'], data['response'])

  print('Recovered Parameters:', recovered_params)

  # 3. next is to use fitted parameters and simulate again
  transmit = vrc.Transmitter(symbols,
                             recovered_params['signal_freq'],
                             recovered_params['snr'] / recovered_params['signal_freq'],
                             recovered_params['inference_freq'],
                             recovered_params['decision_threshold'],
                             recovered_params['timeout_in_sec'])

  data[['recovered_response', 'recovered_rt']] = \
      data['stimulus'].apply(transmit).apply(pd.Series)

  recovered_accuracy = \
      data.query('stimulus == recovered_response')['recovered_rt'].count() / len(data)

  print('medians (true, recovered):',
        data['rt'].median(), data['recovered_rt'].median())

  # TODO use pytest to define the output path
  data.to_csv('outputs/test_reports/test_param_recovery-simulations.csv')

  data = data.melt(value_vars=['rt', 'recovered_rt'],
                   var_name='kind',
                   value_name='rt_value')

  # 4. finally plot ground truth RTs vs recovered RTs
  _, ax = plt.subplots(1, 1)
  binwidth = 1. / inference_freq
  sns.histplot(data,
               x='rt_value', hue='kind',
               binwidth=binwidth, kde=True, label="RT (s)")

  ax.set(xlabel=('Response time (s)\n'
                 f'Recovered: {recovered_accuracy * 100:.1f} %correct\n'
                 f'Truth: {true_accuracy * 100:.1f} %correct'))

  plt.suptitle('Truth v.s. Recovered distributions')

  if store_ax_report:
    # plot ax diagnosis report
    ax_plot_config = interact_contour(ax_model, 'nll_loss')

    # TODO use pytest to define the output path
    with open('outputs/test_reports/test_param_recovery-optimization_report.html', 'w') as f:
        f.write(render_report_elements('param_recovery_report',
                html_elements=[plot_config_to_html(ax_plot_config)],
                header=False, offline=True))
