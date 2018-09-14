import scipy.io as sio

charset = set()

data = sio.loadmat('testdata.mat')['testdata'][0]
imgs, labs = zip(*[(x[0][0], x[1][0]) for x in data])
with open('anno.txt','wb') as anno:
    for img,lab in zip(imgs,labs):
        img,lab = str(img), str(lab)
        anno.write(img + ' ' + lab + '\n')
        charset.update(lab)
charset = list(charset)
with open('charset.txt','wb') as char:
    for w in charset:
        char.write(w + '\n')


