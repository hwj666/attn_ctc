import glob
import os
import cv2
import xml.etree.ElementTree as ET

save_dir = 'image/'
image_dir = '../'

anno = open('anno.txt','wb')
char = open('charset.txt', 'wb')
charset = set()

tree = ET.parse('../test.xml')
root = tree.getroot()

for img in root.iter('image'):
    img_path = img.find('imageName').text
    print os.path.join(image_dir,img_path)
    image = cv2.imread(os.path.join(image_dir,img_path))
    rects = img.find('taggedRectangles')
    for idx, rect in enumerate(rects.findall('taggedRectangle')):
        x,y = rect.attrib['x'],rect.attrib['y']
        height,width = rect.attrib['height'],rect.attrib['width']
        x,y,height,width = map(int,[x,y,height,width])
        new_img = image[y:y+height,x:x+width,:]
        tag = rect.find('tag').text
        charset.update(list(tag))
        img_name = os.path.basename(img_path)
        img_name,_ = os.path.splitext(img_name)
        img_name = '{}{}_{}.jpg'.format(save_dir,img_name,idx)
        anno.write(img_name + ' ' + tag + '\n')
        cv2.imwrite(img_name, new_img)
charset = list(charset)
for c in charset:
    char.write(c + '\n')
char.close()
anno.close()
