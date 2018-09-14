import os

data_dir = '../data/'
floder = ['ICDAR2003', 'IIIT5K',]

charset = []

with open(data_dir + 'all_anno.txt','wb') as all_anno:
    for dataset_dir in floder:
        
        anno_file = os.path.join(data_dir + dataset_dir,'anno.txt')
        if os.path.isfile(anno_file):
            with open(anno_file,'rb') as anno:
                for sample in anno:
                    all_anno.write(os.path.join(dataset_dir, sample))
        
        charset_file = os.path.join(data_dir + dataset_dir,'charset.txt')
        if os.path.isfile(charset_file):
            with open(charset_file, 'rb') as char:
                for w in char:
                    w = w.strip()
                    if w not in charset:
                        charset.append(w)
                
with open(data_dir + 'all_charset.txt','wb') as all_charset:
    for w in charset:
        all_charset.write(w + '\n')
