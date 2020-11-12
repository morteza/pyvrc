import vrc


def test_noiseless_encoding(symbols, message, noiseless_params):

  signal_freq = noiseless_params['signal_freq']
  noise_freq = noiseless_params['noise_freq']
  timeout_in_sec = noiseless_params['timeout_in_sec']

  # init encoder
  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)

  # generate signal spikes
  spike_trains = encode(message, timeout_in_sec)

  # 1. confirm number of channels: one channel per symbol.
  channels_count = len(spike_trains)
  assert channels_count == len(symbols)

  # 2. message channel must contain spikes if signal_freq > zero.
  assert len(spike_trains[message]) > 0

  # TODO: 3. test if len(spike_trains[message]) ~ signal_freq * timeout_in_sec

  # 4. noisy channels must be empty if noise_freq is zero.
  spike_trains.pop(message)
  assert all([len(s) == 0 for s in spike_trains.values()])


def test_spike_train_plotting(symbols, message, noiseless_params, plt):

  signal_freq = noiseless_params['signal_freq']
  noise_freq = noiseless_params['noise_freq']

  # init encoder
  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)

  # generate signal spikes
  spike_trains = encode(message,
                        noiseless_params['timeout_in_sec'])

  plt.eventplot(spike_trains[message], color="green")

  plt.suptitle('Spike train for the message symbol.', y=0)
