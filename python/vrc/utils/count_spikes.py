import numpy as np
import math


def count_spikes(spike_trains: list,
                 duration: float,
                 counting_freq: float,
                 compress=False) -> np.array:
  """Super-fast multi-channel spike counter.

  It counts at intervall defined by the counting_freq, meaning the output
  has the shape of (channels x (counting_freq*duration + 1))


  TODO: remove redundant columns/timestamps (no spikes in all channels)
  TODO: set column names as timestamps (using numpy.recarrays)
        use this: pd.DataFrame(...).to_records()
        or: np.core.records.fromarrays(a.T,names=[...])
  TODO: [numpy warning] Creating an ndarray from ragged nested sequences is deprecated.
  """

  assert compress is False, "Compressed spike counting is not implemented."

  channels_cnt = len(spike_trains)

  spike_counts = np.zeros((channels_cnt, duration * counting_freq + 1))

  for i, spike_train in enumerate(spike_trains):
    for spike in spike_train:
      spike_counts[i, math.ceil(spike * counting_freq)] += 1
  return np.add.accumulate(spike_counts, axis=1)
