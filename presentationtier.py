import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

import numpy

class PresentationTier(object):
    def __init__(self):
        self.data_fig = plt.figure('data')
        self.CS = None
        # Initialize lock semaphore

    def plot_samples(self, X,Y, x_grid=None, p_grid=None, delta=20):
        classes = numpy.unique(Y)
        color_map = plt.get_cmap('hot') # cm.rainbow
        colors = color_map(numpy.linspace(0, 1, max(classes)+1))
        plt.figure('data')
        plt.clf()
        plt.scatter(X[:,0], X[:,1],
                    color=colors[Y], edgecolor='black')
        plt.legend(classes)

    def update_contourline(self, x_grid, p_grid, delta=20, clabel=False):
        print('Updating contour lines')
        fig = plt.figure('data')
        if self.CS != None:
            for coll in self.CS.collections:
                    coll.remove()
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.CS = plt.contour(x_grid[:,0].reshape(delta,delta),
                         x_grid[:,1].reshape(delta,delta),
                         p_grid.reshape(delta,-1), levels, linewidths=3)
        if clabel == True:
            plt.clabel(self.CS, fontsize=15, inline=2)
        plt.pause(0.00001)
