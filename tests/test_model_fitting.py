import pytest
import vrc

# TODO refactor as fixtures
symbols = list('ABC')
rts = [.5, 1.2, 1.3, .3, 2.2, .8]
stimuli = ['A', 'B', 'B', 'A', 'C', 'A']
timeout_in_sec = 3


def test_ax_backend():

  model = vrc.BayesPoissonModel(symbols, timeout_in_sec, backend='ax')
  best_params = model.fit(rts, stimuli)

  # signal_freq, noise_freq, decision_entropy
  print(best_params)
  assert len(best_params) == 3


def test_scipy_backend():

  with pytest.raises(Exception):
    model = vrc.BayesPoissonModel(symbols, timeout_in_sec)
    best_params = model.scipy_fit(rts, stimuli, )

    # signal_freq, noise_freq, decision_entropy
    print(best_params)
    assert len(best_params) == 3
