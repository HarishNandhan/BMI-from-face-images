# models for face2bmi project

# keras vggface model
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Activation, BatchNormalization
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.utils import load_img, img_to_array

# image manipulation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

# face alignment
from mtcnn.mtcnn import MTCNN

# model metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# common packages
import os
import numpy as np
import pandas as pd

def rmse(x,y):
    return np.sqrt(mean_squared_error(x,y))

def mae(x,y):
    return mean_absolute_error(x,y)

def cor(x,y):
    return pearsonr(x,y)[0]

def auc(label, pred):
    return roc_auc_score(label, pred)
    
def imgs2arr(img_names, img_dir, version = 1):
    """ convert images to mutli-dimensional array
    Args:
        @img_names: image names (e.g. [pic001.png])
        @img_dir: directory of the images (e.g. ./tmp)
        @version: for vggface model preprocessing
    Return:
        np.array
    """
    imgs = []
    for img in img_names:
        imgs += [img2arr(os.path.join(img_dir,img), version)]
    return np.concatenate(imgs)

def process_arr(arr, version):
    """process array (resize, mean-substract)
    Args:
        @arr: np.array
    Return:
        np.array
    """
    img = cv2.resize(arr, (224, 224))
    img = np.expand_dims(img, 0)
    img = utils.preprocess_input(img, version = version)
    return img

def img2arr(img_path, version):
    """convert single image to array
    Args:
        @img_path: full path of the image (e.g. ./tmp/pic001.png)
    Return:
        np.array
    """
    img = load_img(img_path)
    img = img_to_array(img)
    img = process_arr(img, version)
    return img

def crop_img(im,x,y,w,h):
    return im[y:(y+h),x:(x+w),:]

def input_generator(data, bs, img_dir, is_train = True, version = 1):
    """data input pipeline
    Args:
    @data: dataframe
    @bs: batch size
    @img_dir: dir of saved images
    @is_train: train/valid [sample] or test [sequential]
    """
    sex_map = {'Male':1, 'Female':0}
    loop = True
    
    while loop:
        if is_train:
            sampled = data.sample(bs)
            x = imgs2arr(sampled['index'],img_dir, version)
            y = [sampled['bmi'].values, sampled['age'].values, sampled['sex'].map(lambda i: sex_map.get(i,0)).values]
            res = (x,y)
        else:
            if len(data) >= bs:
                sampled = data.iloc[:bs,:]
                data = data.iloc[bs:,:]
                res = imgs2arr(sampled['index'],img_dir, version)
            else: 
                loop = False
        yield res 

class FacePrediction(object):
    
    def __init__(self, img_dir, model_type = 'vgg16', sex_thresh = 0.05):
        self.model_type = model_type
        self.img_dir = img_dir
        self.detector = MTCNN()
        self.sex_thresh = sex_thresh
        if model_type in ['vgg16','vgg16_fc6']:
            self.version = 1
        else:
            self.version = 2
    
    def define_model(self, hidden_dim = 128, drop_rate=0.0, freeze_backbone = True):
        
        if self.model_type == 'vgg16_fc6':
            vgg_model = VGGFace(model = 'vgg16', include_top=True, input_shape=(224, 224, 3))
            last_layer = vgg_model.get_layer('fc6').output
            flatten = Activation('relu')(last_layer)
        else:
            vgg_model = VGGFace(model = self.model_type, include_top=False, input_shape=(224, 224, 3))
            last_layer = vgg_model.output
            flatten = Flatten()(last_layer)
        
        if freeze_backbone:
            for layer in vgg_model.layers:
                layer.trainable = False
                
        def block(flatten, name):
            x = Dense(hidden_dim, name=name + '_fc1')(flatten)
            x = BatchNormalization(name = name + '_bn1')(x)
            x = Activation('relu', name = name+'_act1')(x)
            x = Dropout(drop_rate)(x)
            x = Dense(hidden_dim, name=name + '_fc2')(x)
            x = BatchNormalization(name = name + '_bn2')(x)
            x = Activation('relu', name = name+'_act2')(x)
            x = Dropout(drop_rate)(x)
            return x
        
        x = block(flatten, name = 'bmi')
        out_bmi = Dense(1, activation='linear', name='bmi')(x)
        
        x = block(flatten, name = 'age')
        out_age = Dense(1, activation='linear', name='age')(x)
        
        x = block(flatten, name = 'sex')
        out_sex = Dense(1, activation = 'sigmoid', name = 'sex')(x)

        custom_vgg_model = Model(vgg_model.input, [out_bmi, out_age, out_sex])
        custom_vgg_model.compile('adam', 
                                 {'bmi':'mae','age':'mae','sex':'binary_crossentropy'},
                                 {'sex': 'accuracy'}, 
                                 loss_weights={'bmi': 0.8, 'age':0.1, 'sex':0.1})

        self.model = custom_vgg_model
        

    def train(self, train_data, valid_data, bs, epochs, callbacks):
        train_gen = input_generator(train_data, bs, self.img_dir, True, self.version)
        valid_gen = input_generator(valid_data, bs, self.img_dir, True, self.version)
        self.model.fit_generator(train_gen, len(train_data) // bs, epochs, 
                                 validation_data = valid_gen, 
                                 validation_steps = len(valid_data) //  bs, 
                                 callbacks=callbacks)
        
    def evaulate(self, valid_data):
        imgs = valid_data['index'].values
        arr = imgs2arr(imgs, self.img_dir, self.version)
        bmi, age, sex = self.model.predict(arr)
        metrics = {'bmi_mae':mae(bmi[:,0], valid_data.bmi.values), 
                   'bmi_cor':cor(bmi[:,0], valid_data.bmi.values), 
                   'age_mae':mae(age[:,0], valid_data.age.values), 
                   'sex_auc':auc(valid_data.gender, sex[:,0])}
        return metrics
        
    def save_weights(self, model_dir):
        self.model.save_weights(model_dir)
        
    def load_weights(self, model_dir):
        self.model.load_weights(model_dir)
    
    def detect_faces(self, face_path, confidence):
        img = load_img(face_path)
        img = img_to_array(img)
        box = self.detector.detect_faces(img)
        box = [i for i in box if i['confidence'] > confidence]
        res = [crop_img(img, *i['box']) for i in box]
        res = [process_arr(i, self.version) for i in res]
        return box, res
    
    def predict(self, img_dir, show_img = False):
        if os.path.isdir(img_dir):
            imgs = os.listdir(img_dir)
            arr = imgs2arr(imgs, img_dir, self.version)
        else:
            arr = img2arr(img_dir, self.version)
        preds = self.model.predict(arr)
        
        if show_img and os.path.isdir(img_dir):
            bmi, age, sex = preds
            num_plots = len(imgs)
            ncols = 5
            nrows = int((num_plots - 0.1) // ncols + 1)
            fig, axs = plt.subplots(nrows, ncols)
            fig.set_size_inches(3 * ncols, 3 * nrows)
            for i, img in enumerate(imgs):
                col = i % ncols
                row = i // ncols
                axs[row, col].imshow(plt.imread(os.path.join(img_dir,img)))
                axs[row, col].axis('off')
                axs[row, col].set_title('BMI: {:3.1f} AGE: {:02.0f} SEX: {:2.1f}'.format(bmi[i,0], age[i,0], sex[i,0]), fontsize = 10)
        
        return preds
    
    def predict_df(self, img_dir):
        assert os.path.isdir(img_dir), 'input must be directory'
        fnames = os.listdir(img_dir)
        bmi, age, sex = self.predict(img_dir)
        res = pd.DataFrame({'img':fnames, 'bmi':bmi[:,0], 'age':age[:,0], 'sex':sex[:,0]})
        res['sex_prob'] = res['sex']
        res['sex'] = res['sex'].map(lambda i: 'Male' if i > self.sex_thresh else 'Female')
        
        return res
    
    def predict_faces(self, img_path, show_img = True, color = "white", fontsize = 12, 
                      confidence = 0.95, fig_size = (16,12)):
        
        assert os.path.isfile(img_path), 'only single image is supported'
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        boxes, faces = self.detect_faces(img_path, confidence)
        preds = [self.model.predict(face) for face in faces]
        
        if show_img:
            # Create figure and axes
            num_box = len(boxes)
            fig,ax = plt.subplots()
            fig.set_size_inches(fig_size)
            # Display the image
            ax.imshow(img)
            ax.axis('off')
            # Create a Rectangle patch
            for idx, box in enumerate(boxes):
                bmi, age, sex = preds[idx]
                box_x, box_y, box_w, box_h = box['box']
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                ax.text(box_x, box_y, 
                        'BMI:{:3.1f}\nAGE:{:02.0f}\nSEX:{:s}'.format(bmi[0,0], age[0,0], 'M' if sex[0,0] > self.sex_thresh else 'F'),
                       color = color, fontsize = fontsize)
            plt.show()
        
        return preds