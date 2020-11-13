"""This notebooks creates an animated Bayes Poisson inference process.

It produces an animated gif stored in the following path:

`outputs/plots/animated_bayes_poisson.gif`

:warning: This is a work in progress.

"""
# %%

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np


gif_path = '../outputs/plots/animated_bayes_poisson.gif'
symbols = list('ABCDE')
freq = 10
timeout_in_sec = 4
frames = freq * timeout_in_sec  # just a good estimate for smooth animation

fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=True)


spikes = np.sort(np.random.uniform(0, timeout_in_sec, (len(symbols), freq)))


axes[0].axis('off')
axes[0].set_title('Sender')
axes[0].annotate('Stimulus', (.8, .8), ha="center", va="center", size=10,
                 bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", lw=2))

axes[1].eventplot([[], [], [], []])
axes[1].set_xlim(0, timeout_in_sec)
axes[1].set_yticks([])

axes[2].plot([1, 2, 4], [2, 1, 1])


def animate(i, spikes):
    spikes = (spikes + i / 10) % timeout_in_sec
    axes[1].cla()
    axes[1].axis('off')
    axes[1].set_title('Channels')
    axes[1].eventplot(spikes)
    axes[1].set_xlim(0, timeout_in_sec)
    axes[1].set_yticks([])


anim = FuncAnimation(fig, animate, frames=frames,
                     repeat_delay=100, fargs=(spikes,))

anim.save(gif_path, writer=PillowWriter(fps=10))
plt.close()
