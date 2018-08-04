from __future__ import division
from PIL import Image
import math
import os
import glob

count = 1

def long_slice(image_path, outdir, count):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0

    #if we are at the end, set the lower bound to be the bottom of the image  
        #set the bounding box! The important bit

    new_width = width * 0.6
    new_height = height * 0.6     
    bbox = (left, upper, new_width, new_height)
    working_slice = img.crop(bbox)
    #save the slice
    working_slice.save(os.path.join(outdir, str(count) + "UL" + ".jpg"))
     
    UR_left = width - new_width
    bbox = (UR_left, upper, width, new_height)
    working_slice = img.crop(bbox)
    #save the slice
    working_slice.save(os.path.join(outdir, str(count) + "UR" + ".jpg"))

    LL_upper = height - new_height
    bbox = (left, LL_upper, new_width, height)
    working_slice = img.crop(bbox)
    #save the slice
    working_slice.save(os.path.join(outdir, str(count) + "LL" + ".jpg"))

    LR_upper = LL_upper
    LR_left = UR_left
    bbox = (LR_left, LR_upper, width, height)
    working_slice = img.crop(bbox)
    #save the slice
    working_slice.save(os.path.join(outdir, str(count) + "LR" + ".jpg"))

    count = count + 1
    return count

if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    for image in glob.glob("Wally Images From Aditya - NO UPPER LEFT/*"):
        print(image)
        count = long_slice(image, os.getcwd(), count)