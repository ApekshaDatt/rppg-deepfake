import numpy as np
import matplotlib.pyplot as plt

def generate_signal_plot(mode):

    # TIME AXIS
    t = np.linspace(0, 10, 300)

    # REAL SIGNAL
    if mode == "REAL":

        signal = (
            np.sin(2 * np.pi * 1.2 * t)
            + 0.15 * np.random.randn(len(t))
        )

        graph_color = "#22c55e"

    # THREAT SIGNAL
    elif mode == "THREAT":

        signal = np.sin(2 * np.pi * 1.0 * t)

        graph_color = "#ef4444"

    # UNCERTAIN SIGNAL
    else:

        signal = (
            0.4 * np.sin(2 * np.pi * 1.1 * t)
            + 0.35 * np.random.randn(len(t))
        )

        graph_color = "#eab308"

    # CREATE GRAPH
    fig, ax = plt.subplots(figsize=(5,2))

    ax.plot(t, signal, color=graph_color)

    ax.set_facecolor("#111827")

    fig.patch.set_facecolor("#111827")

    ax.tick_params(colors='white')

    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')

    ax.set_title(
        "Biological Pulse Signal",
        color='white'
    )

    return fig