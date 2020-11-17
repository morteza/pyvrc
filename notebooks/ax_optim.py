# %%
import ax
from ax.plot.contour import interact_contour
# from ax.plot.render import plot_config_to_html
# from ax.utils.report.render import render_report_elements
from ax.utils.notebook.plotting import render, init_notebook_plotting


def cost_func(params):
  x = params['signal_freq']
  y = params['noise_freq']
  loss = abs(x * y - 25)
  print('loss:', loss)
  return loss


parameters_space = [
    {
        'name': 'signal_freq',
        'type': 'range',
        'value_type': 'float',
        'bounds': [-100.0, 100.0]
    },
    {
        'name': 'noise_freq',
        'type': 'range',
        'value_type': 'float',
        'bounds': [-100.0, 100.0]
    }
]

best_params, values, experiment, model = ax.optimize(
    parameters=parameters_space,
    evaluation_function=cost_func,
    objective_name='loss',
    minimize=True,
    total_trials=20
)

ax_plot_config = interact_contour(model, 'loss')

init_notebook_plotting()

render(ax_plot_config)

best_params
