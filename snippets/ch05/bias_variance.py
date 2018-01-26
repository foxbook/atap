# Graphs the bias-variance trade-off.
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('white')
sns.set_context('poster')
sns.set_palette('dark')

X = np.linspace(-1, 1, 100)
Yvar = 5 ** X
Ybias = (1/5) ** X
Yerr = Ybias + Yvar

fig = plt.figure(figsize=(9,6))
axe = fig.add_subplot(111)

axe.plot(X, Yerr, '-', color="#222222", label='Total Error')
axe.plot(X, Ybias, '-', color="#c0392b", label="Bias")
axe.plot(X, Yvar, '-', color="#2980b9", label="Variance")

plt.axvline(x=0, color='#666666', linestyle='--')
axe.text(-0.05, 4, "Target Model Complexity", ha="center", va="center", rotation=90, size=16)

plt.legend(loc='upper right')

axe.get_xaxis().set_ticks([])
axe.get_yaxis().set_ticks([])

axe.set_ylabel('Error')
axe.set_xlabel('Complexity (from Underfit to Overfit)')

fname = "atap_ch04_bias_variance_tradeoff.png"
path = os.path.join(os.path.dirname(__file__), "..", "..", "images", "ch04", fname)

plt.savefig(path)
