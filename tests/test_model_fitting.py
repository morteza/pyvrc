import pytest
import vrc


def test_ax_backend(symbols):

  # sample data (TODO: refactor as fixture)
  rts = [.5, 1.2, 1.3, .3, 2.2, .8]
  stimuli = list('ABBACA')
  timeout_in_sec = 3

  model = vrc.BayesPoissonModel(symbols,
                                timeout_in_sec,
                                backend='ax', ax_total_trials=5)
  best_params = model.fit(rts, stimuli)

  # signal_freq, noise_freq, decision_entropy
  print(best_params)
  assert len(best_params) == 4
