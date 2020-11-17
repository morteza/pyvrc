import logging

from dataclasses import dataclass

import random
import numpy as np
from pandas.core.arrays.sparse import dtype
from scipy import stats
import pandas as pd
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
  max_signal_freq = 100.0
  max_noise_freq = 100.0
  initial_entropy = 2.0  # TODO use a more intelligent bound for entrpy
  fit_only_correct_responses = False  # set True to fit response classes separately
  backend: 'vrc.OptimizerBackend' = 'ax'  # see 'vrc.enums' for available backends

  # Additional parameters passed to Ax.
  # TODO use **kwargs instead
  ax_total_trials: int = 20
  constraints = ['signal_freq + noise_freq <= 150']

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

    response_times = np.array(response_times)

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
    sim = pd.DataFrame({
        'msgs_true': random.choices(self.symbols, k=self.simulations_count)
    })

    # sim['msgs_pred'], sim['rts_pred'] = (np.vectorize(transmit)(sim.msgs_true))
    sim[['msgs_pred', 'rts_pred']] = sim.msgs_true.apply(transmit).apply(pd.Series)

    # check timeouts
    n_timeouts = sim['msgs_pred'].isna().sum()
    print('timed-outs:', n_timeouts)
    if n_timeouts == self.simulations_count:
      return sim, None

    if return_correct_dist:
      # keep only correct transmissions
      correct_sim = sim.query('msgs_true == msgs_pred')

      try:
        kde = stats.gaussian_kde(correct_sim.rts_pred)
      except Exception:
        kde = None

      return sim, kde

    # otherwise, when return_correct_dist=False, create dists for all response classes

    # cm = confusion_matrix(sim.query('msgs_pred.notna()').msgs_true,
    #                       sim.query('msgs_pred.notna()').msgs_pred,
    #                       labels=self.symbols)
    # print(cm)

    dists = {}
    for si in self.symbols:
      for sj in self.symbols:
        rts = sim.query('(msgs_pred == @si) & (msgs_true == @sj)')['rts_pred'].values

        try:
          dists[(si, sj)] = stats.gaussian_kde(rts)
        except Exception:
          dists[(si, sj)] = None

    return sim, dists

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

    sim, dist = self.simulate(**params, return_correct_dist=self.fit_only_correct_responses)

    if isinstance(dist, dict):
      probs = []
      msgs_pred = sim.msgs_pred.to_list()
      for si, sj in zip(msgs_pred, stimuli):
        logp = 0.0  # -np.inf
        if (si is not None) and (dist[(si, sj)] is not None):
          rts = np.array(response_times, dtype='float64')[np.array(stimuli) == si]
          logp = dist[(si, sj)].logpdf(rts)
        probs.append(np.nansum(logp))
    else:
      if dist is None:
        return 0.0  # -inf
      probs = np.vectorize(dist.logpdf)(response_times)

    _nll = - np.sum(probs)

    print('NLL Loss:', _nll)

    return _nll
