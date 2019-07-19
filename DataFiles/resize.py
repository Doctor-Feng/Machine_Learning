#!/usr/bin/python
import os, glob
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
path = raw_input("path:")
width =int(raw_input("the width U want:"))
height =int(raw_input("the height U want:"))
imgslist = glob.glob(path+'/*.*')
#format = raw_input("format:")
format = "jpg"
def small_img():
    for imgs in imgslist:
        imgspath, ext = os.path.splitext(imgs)
        img = Image.open(imgs)
        (x,y) = img.size
        #height =int( y * width /x )
        small_img =img.resize((width,height),Image.ANTIALIAS)
        small_img.save(imgspath +".thumbnail."+format)
        print "done"
if __name__ == '__main__':
    small_img()

