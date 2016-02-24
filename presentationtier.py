import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

import numpy

class PresentationTier(object):
    def __init__(self):
        self.CS = None
        # Initialize lock semaphore

    def plot_samples(self, X,Y, x_grid=None, p_grid=None, delta=20):
        classes = numpy.unique(Y)
        color_map = plt.get_cmap('hot') # cm.rainbow
        colors = color_map(numpy.linspace(0, 1, max(classes)+1))

        self.fig_data = plt.figure('data')
        plt.clf()
        plt.scatter(X[:,0], X[:,1],
                    color=colors[Y], edgecolor='black')
        #plt.legend(classes)
        return self.fig_data

    def update_contourline(self, x_grid, p_grid, delta=20, clabel=False):
        print('Updating contour lines')
        self.fig_data = plt.figure('data')
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
        return self.fig_data

    def reliability_diagram(self, prob, Y, marker='--', label=''):
        hist_tot = numpy.histogram(prob, bins=numpy.linspace(0,1,11))
        hist_pos = numpy.histogram(prob[Y == 1], bins=numpy.linspace(0,1,11))
        plt.plot([0,1],[0,1], 'r--')
        centroids = [numpy.mean(numpy.append(prob[(prob > hist_tot[1][i]) * (prob <
            hist_tot[1][i+1])], hist_tot[1][i]+0.05)) for i in range(len(hist_tot[1])-1)]
        plt.plot(centroids, numpy.true_divide(hist_pos[0]+1,hist_tot[0]+2),
                 marker, linewidth=2.0, label=label)

    def plot_reliability_diagram(self, scores_set, labels_set, legend_set):
        self.fig_reliability = plt.figure('reliability_diagram')
        plt.clf()
        plt.title('Reliability diagram')
        for (scores, labels, legend) in zip(scores_set, labels_set, legend_set):
            self.reliability_diagram(scores, labels, marker='x-', label=legend)
        plt.legend(loc='lower right')
        plt.grid(True)
        return self.fig_reliability

    def plot_accuracy(self, accuracy_train, accuracy_val):
        self.fig_acc = plt.figure('accuracy')
        plt.clf()
        plt.title("Accuracy")
        plt.plot(accuracy_train, 'x-', label='train')
        plt.plot(accuracy_val, '+-', label='val')
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.draw()
        return self.fig_acc

    def plot_error(self, error_train, error_val, ylabel):
        self.fig_error = plt.figure('error')
        plt.clf()
        plt.title(ylabel)
        plt.plot(error_train, 'x-', label='train')
        plt.plot(error_val, '+-', label='val')
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('epoch')
        plt.grid(True)
        plt.draw()
        return self.fig_error

    def plot_histogram_scores(self, scores_train, scores_val):
        self.fig_hist = plt.figure('histogram_scores')
        plt.clf()
        plt.title('Histogram of scores (train)')
        plt.hist([scores_train, scores_val], bins=numpy.linspace(0,1,11))
        plt.grid(True)
        plt.draw()
        return self.fig_hist
