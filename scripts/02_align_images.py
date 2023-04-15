import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

from mtcnn.mtcnn import MTCNN
import cv2
import os
from keras.preprocessing import image
from keras_vggface import utils
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
from matplotlib import pyplot as plt


test_dir = './data/test/single_face/'
train_dir = './data/face/'
test_processed_dir = './data/test/test_aligned/'
train_processed_dir = './data/face_aligned/'

logging.info('list images in test_dir[{}]'.format(test_dir))
os.listdir(test_dir)

detector = MTCNN()

def crop_img(im,x,y,w,h):
    return im[y:(y+h),x:(x+w),:]

def detect_face(face_path):
    img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
    box = detector.detect_faces(img)[0]
    return box

def detect_faces(face_path):
    #img = cv2.cvtColor(cv2.imread(face_path), cv2.COLOR_BGR2RGB)
    img = image.load_img(face_path)
    img = image.img_to_array(img)
    box = detector.detect_faces(img)
    return box

# ## Process Cropping For All test Faces
logging.info('Processing test_dir[{}]'.format(test_dir))

if os.path.exists(test_processed_dir):
    shutil.rmtree(test_processed_dir)
os.mkdir(test_processed_dir)

for img in tqdm(os.listdir(test_dir)):
    box = detect_face(test_dir+img)
    im = plt.imread(test_dir+img)
    cropped = crop_img(im, *box['box'])
    plt.imsave(test_processed_dir+img, crop_img(im, *box['box']))


# ## Process Cropping For All train Faces
logging.info('Processing train_dir[{}]'.format(train_dir))

def cut_negative_boundary(box):
    res = []
    for x in box['box']:
        if x < 0:
            x = 0
        res.append(x)
    box['box'] = res
    return box

if os.path.exists(train_processed_dir):
    shutil.rmtree(train_processed_dir)
os.mkdir(train_processed_dir)

for img in tqdm(os.listdir(train_dir)):
    try:
        box = detect_face(train_dir+img)
        box = cut_negative_boundary(box)
        im = plt.imread(train_dir+img)
        cropped = crop_img(im, *box['box'])
        plt.imsave(train_processed_dir+img, cropped)
    except:
        logging.WARN('unable to process image [{:d}.jpg]'.format(img))
        continue


