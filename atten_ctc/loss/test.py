#%%
import matplotlib.pyplot as plt
import glob
import os
files = glob.glob('*.txt')
legend = []
for txt in files:
    name,_ = txt.split('.')
    legend.append(name)
    with open(txt,'r') as lines:
        lines = map(lambda l: float(l.strip()), lines)
    plt.plot(lines)

plt.legend(legend)
plt.show()