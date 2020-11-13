from dataclasses import dataclass

import random

import numpy as np
import scipy
from scipy import stats

import ax

from . import Transmitter


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
  inference_freq: float = 100
  min_signal_freq = 1.0
  min_noise_freq = 0.1
  max_signal_freq = 100.0
  max_noise_freq = 99.0
  initial_entropy = 2.0  # TODO use a more intelligent bound for entrpy
  backend: str = 'ax'  # or 'scipy' (DEPRECATED)

  def fit(self, response_times, stimuli):
    """Use Scipy to fit a ML model.

    Args:
    response_times (list):
      list of all response times in float type.
    stimuli (list):
      list of all stimuli, one per trial. Each stimulus must be a single
      character from the list of symbols.
    backend (str, optional):
      defines the underlying optimization toolkit. Either 'ax' or 'scipy'.
      Default is 'scipy' which uses scipy.optimize.minimize(...) function.
    """
    if self.backend == 'ax':
      return self.ax_fit(response_times, stimuli)

    return self.scipy_fit(response_times, stimuli)

  def ax_fit(self, response_times, stimuli):

    ax_params = [{
        "name": "signal_freq",
        "type": "range",
        # "value_type": "int",  # to speed up optimization
        "bounds": [self.min_signal_freq, self.max_signal_freq]
    }, {
        "name": "noise_freq",
        "type": "range",
        # "value_type": "int",  # to speed up optimization
        "bounds": [self.min_noise_freq, self.max_noise_freq]
    }, {
        "name": "decision_entropy",
        "type": "range",
        "bounds": [0.1, self.initial_entropy]
    }]

    def neg_log_likelihood(p):
      """Negative log-likelihood evaluation function to be minimized."""

      simulated_dist = self.simulate(p['signal_freq'],
                                     p['noise_freq'],
                                     p['decision_entropy'])

      # vectorize logpdf() and calculate total negative-log-likelihood
      vlogpdf = np.vectorize(simulated_dist.logpdf)
      neg_ll = - np.sum(vlogpdf(response_times))
      return neg_ll

    best_params, best_vals, expriment, model = ax.optimize(
        parameters=ax_params,
        evaluation_function=neg_log_likelihood,
        minimize=True,
        objective_name='neg_log_likelihood',
        parameter_constraints=['signal_freq + noise_freq <= 1000.0']
    )

    return best_params

  def scipy_fit(self, response_times, stimuli):
    """[DEPRECATED] Use Scipy to fit a MLE.

    deprecated:
      Scipy optimizer is deprecated. Use Pytorch-based `fit()` instead.

    Args:
    response_times (list):
      list of all response times in float type.
    stimuli (list):
      list of all stimuli, one per trial. Each stimulus must be a single
      character from the list of symbols.
    """

    # TODO: set 'ftol' (or 'maxiter') to reasonable values
    scipy_options = {'disp': True}
    # DEBUG scipy_options = {'disp': True, 'maxiter': 10, 'eps': 0.1}

    # TODO: use better guesses
    # signal_rate, noise_rate, decision_entropy
    initial_guess = [
        self.max_signal_freq / 2,
        self.max_noise_freq / 2,
        self.initial_entropy / len(self.symbols)]

    model = scipy.optimize.minimize(self.neg_log_likelihood,
                                    initial_guess,
                                    method='L-BFGS-B',
                                    options=scipy_options,
                                    args=(response_times, stimuli))

    param_names = ['signal_freq', 'noise_freq', 'decision_entropy']
    best_params = model.x
    return dict(zip(param_names, best_params))

  def simulate(self,
               signal_freq,
               noise_freq,
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
    decision_entropy (float):
      Decision thereshold in bits.

    Returns:
    --------
    A Gaussian KDE distribution with a logpdf(...) function.
    """
    transmit = Transmitter(self.symbols,
                           signal_freq,
                           noise_freq,
                           self.inference_freq,
                           decision_entropy,
                           self.timeout_in_sec,
                           decoder_type='snr')

    # generate a random sequence of messages and record transmission times
    msgs = random.choices(self.symbols, k=self.simulations_count)
    accuracies, transmission_times = np.vectorize(transmit)(msgs)

    # keep only valid transmissions
    transmission_times = transmission_times[accuracies == 1.0]

    if transmission_times.shape[0] == 0:
      # all simulations are failed
      return stats.uniform(0, self.timeout_in_sec)
    elif np.all(transmission_times == transmission_times[0]):
      # sanity check to avoid singular matrix which causes kde bug
      return stats.uniform(0, self.timeout_in_sec)
    else:
      dist = stats.gaussian_kde(transmission_times)
      return dist
