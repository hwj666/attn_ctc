import os

all_anno = open('anno.txt', 'wb')
charset = set()

for src_dir in ['svt_test','svt_train', 'ICDAR2003', 'IIIT5K_test','IIIT5K_train']:
    anno_path = os.path.join(src_dir,'anno.txt')
    char_path = os.path.join(src_dir,'charset.txt')
    with open(anno_path,'rb') as anno:
        for sample in anno:
            all_anno.write(src_dir + '/' + sample)
    
    with open(char_path,'rb') as char:
        charset.update(char.readlines())
charset = list(charset)
with open('charset.txt','wb') as f:
    f.write(''.join(charset))
all_anno.close()