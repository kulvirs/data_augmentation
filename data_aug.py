"""
This program augments (performs rotate/shear operations) on all images with the .jpg extension in the specified
folder (the extension type can be changed to .png below) and puts them in a folder called augmented.
If no file path is given as an argument when running the program, it will just augment images in the current working directory.
I used this script to generate extra images for my Data Mining project so that we would have a larger dataset to work with.
Designed for Python 3.6.
"""

from skimage import transform as tf
from skimage import io
import numpy as np
import math
import sys
import os
import random

DIR_NAME = 'augmented' #name of the folder augmented images are placed in
FILL_MODE = 'wrap' #options: constant, edge, symmetric, reflect, wrap
MAX_ROTATE_ANGLE =  math.pi/float(6) #radians
MAX_SHEAR_ANGLE = 0.25 #radians\
IMG_TYPE = ".jpg" #image extension type

def rotate(name,img_arr):
    rotation_angle = random.uniform(0,MAX_ROTATE_ANGLE)
    rotate_tf = tf.SimilarityTransform(rotation=rotation_angle)
    rotated = tf.warp(img_arr,rotate_tf,mode=FILL_MODE)
    back_rotated = tf.warp(img_arr,rotate_tf.inverse,mode=FILL_MODE)
    io.imsave(name+"_rotated" + IMG_TYPE,rotated)
    io.imsave(name+"_backrotated" + IMG_TYPE,back_rotated)

def shear(name,img_arr):
    shear_angle = random.uniform(0,MAX_SHEAR_ANGLE)
    affine_tf = tf.AffineTransform(shear=shear_angle,rotation=-shear_angle)
    sheared = tf.warp(img_arr,affine_tf,mode=FILL_MODE)
    sheared_inv = tf.warp(img_arr,affine_tf.inverse,mode=FILL_MODE)
    io.imsave(name+"_sheared" + IMG_TYPE,sheared)
    io.imsave(name+"_sheared_inverse" + IMG_TYPE,sheared_inv)

def flip(name,img_arr):
    flipped = np.fliplr(img_arr)
    io.imsave(name+"_flipped"+IMG_TYPE,flipped)
    return flipped

def swirl(name,img_arr):
    swirled = tf.swirl(img_arr,mode=FILL_MODE)
    io.imsave(name+"_swirled"+IMG_TYPE,swirled)

def main():
    if len(sys.argv) == 2:
        os.chdir(sys.argv[1])
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    for filename in os.listdir(os.getcwd()):
        file_name_ext = os.path.splitext(filename)
        if file_name_ext[1] == IMG_TYPE:
            img_arr = io.imread(filename)
            aug_filename = DIR_NAME + "/" + file_name_ext[0]
            io.imsave(aug_filename+IMG_TYPE,img_arr)
            rotate(aug_filename,img_arr)
            swirl(aug_filename,img_arr)
            shear(aug_filename,img_arr)
            flipped = flip(aug_filename,img_arr)
            rotate(aug_filename+"_flipped",flipped)
            shear(aug_filename+"_flipped",flipped)
            swirl(aug_filename+"_flipped",flipped)

if __name__ == "__main__":
    main()
