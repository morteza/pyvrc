import pytest
import vrc


def test_noiseless_encoding(symbols, stimulus, noiseless_params):

  signal_freq = noiseless_params['signal_freq']
  noise_freq = noiseless_params['noise_freq']
  timeout_in_sec = noiseless_params['timeout_in_sec']

  # init encoder
  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)

  # generate signal spikes
  spike_trains = encode(stimulus, timeout_in_sec)

  # 1. confirm number of channels: one channel per symbol.
  channels_count = len(spike_trains)
  assert channels_count == len(symbols)

  # 2. stimulus channel must contain spikes if signal_freq > zero.
  assert len(spike_trains[stimulus]) > 0

  # TODO: 3. test if len(spike_trains[stimulus]) ~ signal_freq * timeout_in_sec

  # 4. noisy channels must be empty if noise_freq is zero.
  spike_trains.pop(stimulus)
  assert all([len(s) == 0 for s in spike_trains.values()])


@pytest.mark.parametrize('params_fixture', ['noisy_params', 'noiseless_params'])
def test_spike_train_plotting(symbols, stimulus, params_fixture, plt, request):

  params = request.getfixturevalue(params_fixture)

  signal_freq = params['signal_freq']
  noise_freq = params['noise_freq']
  timeout_in_sec = params['timeout_in_sec']

  # init encoder
  encode = vrc.OneHotEncoder(symbols, signal_freq, noise_freq)

  # generate signal spikes
  spike_trains = encode(stimulus, timeout_in_sec)

  plt.eventplot(spike_trains.values(), color="green")
  plt.suptitle('Spike train for the stimulus', y=0)
