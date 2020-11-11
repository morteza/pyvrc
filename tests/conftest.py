import pytest
from pathlib import Path


@pytest.fixture
def outputs_dir():
  """Provides path to the outputs/ directory.
  """

  d = 'outputs/'
  assert Path(d).exists()
  return d
