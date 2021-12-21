
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

f, ax = plt.subplots(1)  #
f.set_figheight(7)
f.set_figwidth(23)
#ax.set_facecolor('w')


sns.lineplot(np.arange(450), np.zeros(450), ax=ax)

ax.axvspan(0, 50, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][0], alpha=0.35)
ax.axvspan(50, 100, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][1], alpha=0.35)
ax.axvspan(100, 150, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][2], alpha=0.35)
ax.axvspan(150, 200, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][3], alpha=0.35)
ax.axvspan(200, 250, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][4], alpha=0.35)
ax.axvspan(250, 300, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][5], alpha=0.35)
ax.axvspan(350, 400, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][6], alpha=0.35)
ax.axvspan(400, 450, facecolor=['g', 'r', 'b', 'y', 'm', 'c', 'k', 'w'][7], alpha=0.35)


plt.savefig('test.pdf')
