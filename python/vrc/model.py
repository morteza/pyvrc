from dataclasses import dataclass

import random
import numpy as np
# from scipy import stats
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
import ax

import vrc


@dataclass
class BayesPoissonModel():
  """Bayes Poisson Variable Rate Coding model.

  Args:
  -----
  symbols (list):
    list of all possible stimuli.
  timeout_in_sec (float):
    timeout in seconds.
  n_simulations (int):
    Number of simulations in each step of optimization.
  """

  symbols: list
  timeout_in_sec: float
  n_simulations: int = 100

  backend: 'vrc.OptimizerBackend' = 'ax'  # see 'vrc.enums' for available backends

  print_confusion_matrix = False

  # Additional parameters (TODO: move to kwargs).
  min_signal_freq = 1.0
  max_signal_freq = 200.0
  min_snr = 0.01
  max_snr = 100.0
  min_decision_threshold = 0.1
  initial_entropy = 1.0  # will be updated upon knowing all possible symbols
  min_inference_freq = 50
  max_inference_freq = 150
  min_kde_bandwidth = 0.1
  max_kde_bandwidth = 10.0

  ax_total_trials = 20
  ax_constraints = []  # ['signal_freq + noise_freq <= 200']

  def get_parameters_space(self):
    return [
        {
            'name': 'signal_freq',
            'type': 'range',
            'value_type': 'float',
            'bounds': [self.min_signal_freq, self.max_signal_freq]
        },
        {
            'name': 'snr',
            'type': 'range',
            'value_type': 'float',
            'bounds': [self.min_snr, self.max_snr]
        },
        {
            'name': 'decision_threshold',
            'type': 'range',
            'value_type': 'float',
            'bounds': [self.min_decision_threshold, self.initial_entropy]
        },
        {
            'name': 'inference_freq',
            'type': 'range',
            'value_type': 'int',
            'bounds': [self.min_inference_freq, self.max_inference_freq],
            # 'type': 'fixed',
            # 'value': self.min_inference_freq
        },
        {
            'name': 'kde_bandwidth',
            'type': 'range',
            'value_type': 'float',
            'bounds': [self.min_kde_bandwidth, self.max_kde_bandwidth]
        },
        {
            'name': 'timeout_in_sec',
            'type': 'fixed',
            'value_type': 'float',
            'value': self.timeout_in_sec
        }
    ]

  def fit(self, response_times, stimuli):
    """Use the specified backend to fit a MLE model.

    Args:
    response_times (list):
      list of all response times in float type, one per tria.
    stimuli (list):
      list of all stimuli, one per trial. Each stimulus must be a single
      character from the list of symbols.
    backend (str, optional):
      defines the underlying optimization toolkit.
      Default is 'ax' which uses PyTorch Bayesian Optimizer.
    """

    if self.backend == vrc.OptimizerBackend.AX.value:
      best_params, *_ = self.ax_fit(response_times, stimuli)
      return best_params

    raise NotImplementedError(f'{self.backend} is not implemented yet.')

  def ax_fit(self, response_times, stimuli):

    valid_rts = np.isfinite(response_times)

    stimuli = stimuli[valid_rts]
    stimuli = np.array(stimuli, dtype='str')

    response_times = response_times[valid_rts]
    response_times = np.array(response_times, dtype='float64')

    best_params, best_vals, experiment, model = ax.optimize(
        parameters=self.get_parameters_space(),
        evaluation_function=(lambda params: self.nll_loss(params, response_times, stimuli)),
        minimize=True,
        objective_name='nll_loss',
        parameter_constraints=self.ax_constraints,
        total_trials=self.ax_total_trials
    )

    return best_params, best_vals, experiment, model

  def simulate(self, params):
    """Simulate multi-channel one-hot VRC transmission and return RTs.

    Args:
    -----
    params (dict):
      signal_freq (float):
        Signal frequency in Hz.
      noise_freq (float):
        Noise frequency in Hz.
      inference_freq (float):
        Inference frequency.
      decision_threshold (float):
        Decision thereshold for the remaining entropy in bits.

    Returns:
    --------
    A dataframe containing simulated RTs and information about simulated stimuli
    and responses.
    """

    noise_freq = params['signal_freq'] / params['snr']

    transmit = vrc.Transmitter(symbols=self.symbols,
                               signal_freq=params['signal_freq'],
                               noise_freq=noise_freq,
                               inference_freq=params['inference_freq'],
                               decision_threshold=params['decision_threshold'],
                               timeout_in_sec=params['timeout_in_sec'],
                               decoder_type=vrc.DecoderType.SNR.value)

    # generate a random sequence of stimuli and record response times
    sim = pd.DataFrame({
        'stimulus': random.choices(self.symbols, k=self.n_simulations)
    })

    sim[['response', 'rt']] = sim['stimulus'].apply(transmit).apply(pd.Series)

    if self.print_confusion_matrix:
      cm_sim = sim.fillna(' ')
      cm = confusion_matrix(cm_sim.stimulus.values,
                            cm_sim.response.values,
                            labels=self.symbols)
      print(cm)

    return sim

  def nll_loss(self, params, response_times, stimuli):
    """Negative log-likelihood loss function to be minimized.

    Args:
    -----
    params (dict):
      Simulation hyperparameters.
    """

    sim = self.simulate(params)  # (**params)

    log_probs = []

    for r in self.symbols:  # response
      for s in self.symbols:  # stimulus
        true_rts = response_times[stimuli == s]
        sim_rts = sim.query('(stimulus == @s) & (response == @r)')['rt']

        # only keep finite simulated RTs
        sim_rts = sim_rts[np.isfinite(sim_rts)].values

        try:
          # kd = stats.gaussian_kde(sim_rts)
          # log_p = np.sum(kd.logpdf(true_rts))

          bw = params['kde_bandwidth']
          sim_rts = sim_rts.reshape(-1, 1)
          true_rts = true_rts.reshape(-1, 1)
          kd = KernelDensity(kernel='gaussian', bandwidth=bw)
          kd.fit(sim_rts)
          log_p = kd.score(true_rts)
          # vrc.utils.plot_kde(kd, true_rts)
        except Exception:
          log_p = np.nan

        log_probs.append(log_p)

    _nll = - np.nansum(log_probs)
    print('nll_loss:', _nll)

    return _nll
