import logging

from dataclasses import dataclass

import random
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
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
  fit_only_correct_responses = False  # set True to fit response classes separately
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
        evaluation_function=(lambda params: self.nll_loss(params, response_times, stimuli)),
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
               decision_entropy,
               return_correct_dist=False):
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
    return_correct_dist (boolean):
      Only return distribution for the correct responses. Otherwise, a set on distributions
      will be returned; one per predicted class.

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
    pred_msgs, pred_rts = np.vectorize(transmit)(msgs)

    pred_msgs = ['' if x is None else x for x in pred_msgs]

    if return_correct_dist:
      corrects = (pred_msgs == msgs)

      print('accuracy = {}'.format(corrects.sum() / len(corrects)))
      # TODO keep all transmissions and fit the full confusion matrix
      # keep only valid transmissions
      pred_rts = pred_rts[corrects]

      try:
        return stats.gaussian_kde(pred_rts)
      except Exception:
        return None

    # return dists for all response classes

    n_timeouts = np.sum([1 if m is None else 0 for m in pred_msgs])
    if n_timeouts == len(pred_msgs):
      return None

    # cm = confusion_matrix(msgs, pred_msgs, labels=self.symbols)
    dists = {}
    for c in self.symbols:
      class_rts = pred_rts[pred_msgs == c]

      try:
        dists[c] = stats.gaussian_kde(class_rts)
      except Exception:
        dists[c] = None

    return dists

  def nll_loss(self, params, response_times, stimuli):
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

    if self.fit_only_correct_responses:
      simulated_dist = self.simulate(**params, return_correct_dist=True)
      if simulated_dist is None:
        return -np.inf
      logpdfs = np.vectorize(simulated_dist.logpdf)(response_times)
    else:
      simulated_dists = self.simulate(**params, return_correct_dist=False)
      logpdfs = [-np.inf
                 if simulated_dists[s] is None
                 else simulated_dists[s].logpdf(response_times[stimuli == s])
                 for s in stimuli]

    _nll = - np.sum(logpdfs)

    return _nll
