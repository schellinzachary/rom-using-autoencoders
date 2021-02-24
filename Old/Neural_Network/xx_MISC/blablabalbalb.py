import numpy as np
import matplotlib.pyplot as plt


# Plot a sine and cosine curve
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')

# Set up grid, legend, and limits
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0, 3 * np.pi);
axr = ax.twiny()


axr.xaxis.set_major_locator(plt.FixedLocator((np.linspace(0,1,25))))
#3axr.axis.set_major_locator(plt.)
#ax.yaxis.set_minor_locator(plt.FixedLocator(0))
#axr.yaxis.set_minor_locator(plt.FixedLocator((0.2,0.3)))


plt.show()

