# -*- coding: utf-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt


# Create a new figure
fig, ax = plt.subplots()

# Create the X data points as a numpy array
X = np.linspace(-10, 10, 255)

# Compute two quadratic functions
Y1 = 2*X ** 2 + 10
Y2 = 3*X ** 2 + 50

ax.plot(X, Y1)
ax.plot(X, Y2)

fig.suptitle('Introduction to Plotting', fontsize=14, fontweight='bold')
ax.set_title('Two Quadratic Functions', color='navy')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
text = '''
       A parabola is a U-shaped, mirror-
       symmetrical plane curve. In this case, 
       the turning points for both parabolae 
       are minima.
       '''
ax.text(0.5, 0.75, text, wrap=True, verticalalignment='top',
        horizontalalignment='center', transform=ax.transAxes,
        fontsize=8)

formula = r'$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$'
plt.annotate(formula, xy=(6, 50))

plt.show()
