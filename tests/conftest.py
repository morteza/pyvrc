import pytest
from pathlib import Path


@pytest.fixture
def outputs_dir():
  """Provides path to the outputs/ directory.
  """

  d = 'outputs/'
  assert Path(d).exists()
  return d


@pytest.fixture
def noiseless_params():
  """Non-noisy Variable Rate Coding hyper-parameters."""

  params = dict(
      signal_freq=10,
      noise_freq=0,
      inference_freq=100,
      timeout_in_sec=1,
      decision_entropy=0.8
  )

  return params


@pytest.fixture
def noisy_params():
  """Noisy Variable Rate Coding hyper-parameters."""

  params = dict(
      signal_freq=100,
      noise_freq=3,
      inference_freq=100,
      timeout_in_sec=1,
      decision_entropy=0.8
  )

  return params


@pytest.fixture
def symbols():
  return list('ABCD')


@pytest.fixture
def stimulus():
  return 'B'
