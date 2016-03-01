import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

import numpy as np

class PresentationTier(object):
    def __init__(self):
        self.CS = None
        # Initialize lock semaphore

    def plot_samples(self, X,Y, x_grid=None, p_grid=None, delta=20):
        classes = np.unique(Y)
        color_map = plt.get_cmap('hot') # cm.rainbow
        colors = color_map(np.linspace(0, 1, max(classes)+1))

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

    def reliability_diagram(self, prob, Y, marker='--', label='', alpha=1,
            linewidth=1):
        '''
            alpha= Laplace correction, default add-one smoothing
        '''
        bins = np.linspace(0,1,11)
        hist_tot = np.histogram(prob, bins=bins)
        hist_pos = np.histogram(prob[Y == 1], bins=bins)
        plt.plot([0,1],[0,1], 'r--')
        # Compute the centroids of every bin
        # FIXME: check if it is incorrect to consider points in the
        # intersection belonging to both bins
        centroids = [np.mean(np.append(
                     prob[np.where(np.logical_and(prob >= bins[i],
                                                  prob <= bins[i+1]))],
                     bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]

        proportion = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+alpha*2)
        self.ax_reliability.plot(centroids, proportion,
                                 marker, linewidth=linewidth, label=label)

    def plot_reliability_diagram(self, scores_set, labels_set, legend_set,
            original_first=False, alpha=1, **kwargs):
        self.fig_reliability = plt.figure('reliability_diagram')
        self.fig_reliability.clf()
        self.ax_reliability = plt.subplot(111)
        ax = self.ax_reliability
        ax.set_title('Reliability diagram')
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        n_lines = len(legend_set)
        if original_first:
            bins = np.linspace(0,1,11)
            hist_tot = np.histogram(scores_set[0], bins=bins)
            hist_pos = np.histogram(scores_set[0][labels_set[0] == 1], bins=bins)
            edges = np.insert(bins, np.arange(len(bins)), bins)
            empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
            empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                    empirical_p)
            ax.plot(edges[1:-1], empirical_p, label='empirical')

        skip = original_first
        for (scores, labels, legend) in zip(scores_set, labels_set, legend_set):
            if skip and original_first:
                skip = False
            else:
                self.reliability_diagram(scores, labels, marker='x-',
                        label=legend, linewidth=n_lines, alpha=alpha, **kwargs)
                n_lines -= 1
        if original_first:
            ax.plot(scores_set[0], labels_set[0], 'kx', label=legend_set[0],
                    markersize=9, markeredgewidth=1)
        ax.legend(loc='lower right')
        ax.grid(True)
        return self.fig_reliability

    def plot_reliability_map(self, scores_set, prob_set, legend_set,
            original_first=False, alpha=1, **kwargs):
        self.fig_reliability_map = plt.figure('reliability_map')
        self.fig_reliability_map.clf()
        self.ax_reliability_map = plt.subplot(111)
        ax = self.ax_reliability_map
        ax.set_title('Reliability map')
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        n_lines = len(legend_set)
        if original_first:
            bins = np.linspace(0,1,11)
            hist_tot = np.histogram(scores_set[0], bins=bins)
            hist_pos = np.histogram(scores_set[0][prob_set[0] == 1], bins=bins)
            edges = np.insert(bins, np.arange(len(bins)), bins)
            empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
            empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                    empirical_p)
            ax.plot(edges[1:-1], empirical_p, label='empirical')

        skip = original_first
        for (scores, prob, legend) in zip(scores_set, prob_set, legend_set):
            if skip and original_first:
                skip = False
            else:
                ax.plot(scores, prob, '-', label=legend,
                          linewidth=n_lines, **kwargs)
                n_lines -= 1
        if original_first:
            ax.plot(scores_set[0], prob_set[0], 'kx', label=legend_set[0],
                    markersize=9, markeredgewidth=1)
        ax.legend(loc='lower right')
        ax.grid(True)
        return self.fig_reliability_map

    def plot_accuracy(self, accuracy_set, legend_set):
        self.fig_acc = plt.figure('accuracy')
        plt.clf()
        plt.title("Accuracy")
        for accuracy, label in zip(accuracy_set, legend_set):
            plt.plot(accuracy, label=label)
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.draw()
        return self.fig_acc

    def plot_error(self, error_set, legend_set, ylabel):
        self.fig_error = plt.figure('error')
        plt.clf()
        plt.title(ylabel)
        for error, label in zip(error_set, legend_set):
            plt.plot(error, label=label)
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel('epoch')
        plt.grid(True)
        plt.draw()
        return self.fig_error

    def plot_histogram_scores(self, scores_set):
        self.fig_hist = plt.figure('histogram_scores')
        plt.clf()
        plt.title('Histogram of scores (train)')
        plt.hist(scores_set, bins=np.linspace(0,1,11))
        plt.grid(True)
        plt.draw()
        return self.fig_hist
