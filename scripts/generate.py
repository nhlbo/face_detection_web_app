from PIL import Image, ImageOps

import numpy as np
import cv2
import random
import glob
import os
import time


character = ['bezos', 'gate', 'mark', 'musk', 'page'] #page 50


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def resizeImage(image):
    scale_percent = random.randrange(55, 65) # percent of original size
    h = 400
    w = 600 
    width = int(image.shape[1] * (h * scale_percent / 100) / image.shape[0])
    height = int(h * scale_percent / 100)
    #width = int(image.shape[1] * scale_percent / 100)
    #height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized
#doSth()

bg_images = [Image.open(file) for file in glob.glob('background/*.jpg')]  #read file background
num = int(input())
face_images = [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in glob.glob('character/' +character[num]+'/*.png')]

for i in range(len(face_images)):
    image = face_images[i]
    image = resizeImage(image)
    angle = random.randrange(5, 10)
    rotated = rotate_bound(image, angle)
    cv2.imwrite("temp.png", rotated)

    #random background picture
    pos = random.randrange(0, 20)
    bg_images[pos].save("back1.png")
    back_image = Image.open("back1.png")

    face_image = Image.open("temp.png")

    pos_xy = (random.randrange(0, 300), random.randrange(0, 120))

    back_image.paste(face_image, pos_xy, face_image)

    back_image.save('final/'+character[num]+'/result' + str(i) + '.jpg')

    #mirror face
    pos = random.randrange(0, 20)
    bg_images[pos].save("back2.png")
    back_image_mirror = Image.open("back2.png")

    face_image_mirror = ImageOps.mirror(face_image)

    back_image_mirror.paste(face_image_mirror, pos_xy, face_image_mirror)
    back_image_mirror.save('final/'+character[num]+'/mirror_result' + str(i) + '.jpg')

    os.remove("temp.png")
    os.remove("back1.png")
    os.remove("back2.png")


   