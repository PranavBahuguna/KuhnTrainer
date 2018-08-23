import matplotlib.pyplot as plt
import numpy as np

def plot(x, y, e, label):
    plt.plot(x, y, label=label)
    plt.fill_between(x, np.subtract(y, e), np.add(y, e), alpha=0.5)

def showPlot(title, xLabel, yLabel, xScaling, yScaling):
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if xScaling:
        plt.xscale(xScaling)
    if yScaling:
        plt.yscale(yScaling)
    plt.grid(True)
    plt.legend()
    plt.show()