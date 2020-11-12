import numpy as np


def count_spikes(spike_trains: np.array,
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
  """

  assert compress is False, "Compressed spike counting is not implemented."

  channels_cnt = spike_trains.shape[0]

  spike_counts = np.zeros((channels_cnt, duration * counting_freq + 1))

  for i, spike_train in enumerate(spike_trains):
    for spike in spike_train:
      spike_counts[i, int(spike * counting_freq)] += 1
  return np.add.accumulate(spike_counts, axis=1)
