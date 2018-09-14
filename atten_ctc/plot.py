#%%
import matplotlib.pyplot as plt
val = {}
lenged = []
with open('hwj.txt') as lines:
    for line in lines:
        print line.strip().split('_')
        model,ratio,epoch,data_dir,ctc,attn = line.strip().split('_')
        
        if ratio not in val:
            val[ratio] = [max(1-float(ctc),1-float(attn))]
        else:
            val[ratio] += [max(1-float(ctc),1-float(attn))]

for key,value in val.items():
    if key != '0':
        plt.plot(value)
        lenged.append(key)

plt.legend(lenged)
plt.show()