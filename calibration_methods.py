import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
# TODO implement a MLP calibration method
#from sklearn.neural_network import MLPClassifier

from presentationtier import PresentationTier

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (5,3.5)

np.random.seed(1234)
_EPSILON=10e-8

# Laplace correction
alpha=0.1

#x = T.ivector('x')  # the data is presented as rasterized images
#y = T.ivector('y')  # the labels are presented as 1D vector of
#
#classifier = MLP(
#    rng=np.random.RandomState(42),
#    input=x,
#    n_in=1,
#    n_hidden=[3],
#    n_out=1
#)

S = np.tile(np.linspace(0.01, 0.99, 10), 10)
Y = np.asarray([0,0,0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,0,0,1,1,
                0,0,0,0,0,0,0,0,1,1,
                1,0,0,0,0,0,1,1,1,1,
                1,0,0,0,0,0,1,1,1,1,
                1,0,0,0,0,0,1,1,1,1,
                1,0,0,0,0,0,1,1,1,1,
                1,0,0,0,0,0,1,1,1,1])

print('Learning Isotonic Regression')
ir = IsotonicRegression(increasing=True, out_of_bounds='clip',
                        y_min=_EPSILON, y_max=(1-_EPSILON))
ir.fit(S, Y)
print('Learning Logistic Regression')
lr = LogisticRegression(C=1., solver='lbfgs')
lr.fit(S.reshape(-1,1), Y)



scores_set = [S, ir.predict(S), lr.predict_proba(S.reshape(-1,1))[:,1]]
labels_set = [Y, Y, Y]
legend = ['Y', 'IR', 'LR']

pt = PresentationTier()
fig = pt.plot_reliability_diagram(scores_set, labels_set, legend,
        original_first=True, alpha=alpha)

scores_lin = np.linspace(0,1,100)
scores_set = [S, scores_lin, scores_lin]
prob_set = [Y, ir.predict(scores_lin),
            lr.predict_proba(scores_lin.reshape(-1,1))[:,1]]
fig = pt.plot_reliability_map(scores_set, prob_set, legend,
        original_first=True, alpha=alpha)
