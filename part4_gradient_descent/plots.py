"""Matplotlib helpers for Part 4. One generic function, reused for every plot."""
import matplotlib.pyplot as plt


def plot_series(series, title, ylabel, filename=None):
    """Line plot of several named series against the iteration number.

    series: {label: 1-D array}. All arrays share the same length.
    """
    plt.figure(figsize=(8, 4.5))
    for label, values in series.items():
        plt.plot(range(len(values)), values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


def plot_parameters(history, filename=None):
    """How m1, m2, b1, b2 change over iterations."""
    series = {
        "m1": history["m"][:, 0], "m2": history["m"][:, 1],
        "b1": history["b"][:, 0], "b2": history["b"][:, 1],
    }
    plot_series(series, "Parameters m and b over iterations", "Parameter value", filename)


def plot_cost(history, filename=None):
    """How the error (MSE) changes over iterations."""
    plot_series({"MSE cost J": history["cost"]}, "Error (MSE) over iterations",
                "Cost J", filename)
