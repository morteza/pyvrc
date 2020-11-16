import logging

from dataclasses import dataclass

import random
import numpy as np
from scipy import stats
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
  simulations_count (int):
    Number of simulations in each step of optimization.
  inference_freq (float):
    Frequency in Hz that represents the time between two consecutive
    inference in receiver's side.
  """

  symbols: list
  timeout_in_sec: float
  simulations_count: int = 100
  inference_freq: int = 100
  min_signal_freq = 1.0
  min_noise_freq = 0.1
  max_signal_freq = 1000.0
  max_noise_freq = 1000.0
  initial_entropy = 2.0  # TODO use a more intelligent bound for entrpy
  backend: 'vrc.OptimizerBackend' = 'ax'  # see 'vrc.enums' for available backends

  # Additional parameters passed to Ax.
  # TODO use **kwargs instead
  ax_total_trials: int = 20
  constraints = ['signal_freq + noise_freq <= 1000']

  def fit(self, response_times, stimuli):
    """Use the specified backend to fit a MLE model.

    Args:
    response_times (list):
      list of all response times in float type.
    stimuli (list):
      list of all stimuli, one per trial. Each stimulus must be a single
      character from the list of symbols.
    backend (str, optional):
      defines the underlying optimization toolkit.
      Default is 'ax' which uses PyTorch Bayesian Optimizer.
    """
    if self.backend == vrc.OptimizerBackend.AX.value:
      return self.ax_fit(response_times, stimuli)

    raise NotImplementedError(f'{self.backend} is not implemented yet.')

  def ax_fit(self, response_times, stimuli):

    # signal_freq = ax.RangeParameter(name="signal_freq",
    #                                 parameter_type=ax.ParameterType.INT,
    #                                 lower=self.min_signal_freq,
    #                                 upper=self.max_signal_freq)
    # noise_freq = ax.RangeParameter(name="noise_freq",
    #                                parameter_type=ax.ParameterType.INT,
    #                                lower=self.min_noise_freq,
    #                                upper=self.max_noise_freq)
    # decision_entropy = ax.RangeParameter(name="decision_entropy",
    #                                      parameter_type=ax.ParameterType.FLOAT,
    #                                      lower=0.1,
    #                                      upper=self.initial_entropy)

    ax_params = [{
        'name': 'signal_freq',
        'type': 'range',
        'value_type': 'float',  # to speed up optimization use 'int'
        'bounds': [self.min_signal_freq, self.max_signal_freq]
    }, {
        'name': 'noise_freq',
        'type': 'range',
        'value_type': 'float',  # to speed up optimization use 'int'
        'bounds': [self.min_noise_freq, self.max_noise_freq]
    }, {
        'name': 'decision_entropy',
        'type': 'range',
        'value_type': 'float',
        'bounds': [0.1, self.initial_entropy]
    }, {
        'name': 'inference_freq',
        'type': 'fixed',  # TODO make inference_freq a model parameter
        'value_type': 'int',
        'value': self.inference_freq
    }]

    best_params, best_vals, expriment, model = ax.optimize(
        parameters=ax_params,
        evaluation_function=(lambda params: self.nll_loss(params, response_times)),
        minimize=True,
        objective_name='nll_loss',
        parameter_constraints=self.constraints,
        total_trials=self.ax_total_trials
    )

    return best_params

  def simulate(self,
               signal_freq,
               noise_freq,
               inference_freq,
               decision_entropy):
    """Simulate multi-channel transmission and return transmission times.

    Args:
    -----
    message (str):
      Stimulus to be passed through channels.
    signal_freq (float):
      Signal frequency in Hz.
    noise_freq (float):
      Noise frequency in Hz.
    inference_freq (float):
      Inference frequency.
    decision_entropy (float):
      Decision thereshold in bits.

    Returns:
    --------
    A Gaussian KDE distribution with a logpdf(...) function.
    """
    transmit = vrc.Transmitter(self.symbols,
                               signal_freq,
                               noise_freq,
                               inference_freq,
                               decision_entropy,
                               self.timeout_in_sec,
                               decoder_type=vrc.DecoderType.SNR)

    # generate a random sequence of messages and record transmission times
    msgs = random.choices(self.symbols, k=self.simulations_count)
    pred_msgs, transmission_times = np.vectorize(transmit)(msgs)
    accuracies = (pred_msgs == msgs)

    # TODO keep all transmissions and fit the full confusion matrix
    # keep only valid transmissions
    transmission_times = transmission_times[accuracies]

    try:
      dist = stats.gaussian_kde(transmission_times)
    except Exception:
      # TODO: discard simulations if accuracy in confusion matrix is low
      logging.warn('cannot initialize Gaussian KDE, applying generic uniform instead.')
      dist = stats.uniform(0, self.timeout_in_sec)

    return dist

  def nll_loss(self, params, response_times):
    """Negative Log-Likelihood loss function (to be minimized).

    Args:
    -----
    params (dict):
      A parameter dictionary with the following keys:
        - signal_freq
        - noise_freq
        - decision_entropy
        - inference_freq

    """

    simulated_dist = self.simulate(**params)

    # vectorize logpdf and calculate total NLL
    vlogpdf = np.vectorize(simulated_dist.logpdf)
    _nll = - np.sum(vlogpdf(response_times))
    return _nll
