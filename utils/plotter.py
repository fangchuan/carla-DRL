import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from baselines.common import plot_util as pu

def plot_result(file):
    dir = os.path.dirname(file)
    results = pu.load_results(file)
    print("total {} results".format(len(results)))

    r = results[0]
    plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(r.progress.total_timesteps, r.progress.eprewmean)
    plt.show()