from utils import *
from kh_tools import *
import models
import imp
imp.reload(models)
from models import ALOCC_Model
from keras.datasets import mnist

from keras.losses import binary_crossentropy
from keras import backend as K

import numpy as np

import matplotlib.pyplot as plt

import cv2
import numpy as np

from landsat_data_loader import LandsatDataLoader

self = ALOCC_Model(dataset_name='mnist', input_height=28,input_width=28)
self.adversarial_model.load_weights('./checkpoint/ALOCC_Model_4.h5')

root = "/QCOLT/QCOLT_DEV_OPS/"
path = root +'/TDS_NOVELTY_DETECTION/EXP_02//nominal_chips/'    
loader = LandsatDataLoader(path)   
X_train = loader.load_data()
X_train = X_train / 255

print("Number of samples loaded =",X_train.shape[0])
print("Dimensions (H,W) = ({},{})".format(X_train.shape[1],X_train.shape[2]))


def resize_img(img,target_edge=500):
    return cv2.resize(img,
                      (target_edge,target_edge),
                      interpolation = cv2.INTER_AREA)

def test_reconstruction(index=0, show_images=False, res_edge = 28):
    data = X_train[index].reshape(-1, res_edge, res_edge, 1)[0:1]

    model_predicts = self.adversarial_model.predict(data)
    input_image = data[0].reshape((res_edge, res_edge))
    reconstructed_image = model_predicts[0].reshape((res_edge, res_edge))
    
    y_true = K.variable(reconstructed_image)
    y_pred = K.variable(input_image)
    error = K.eval(binary_crossentropy(y_true, y_pred)).mean()
    print(index,'Reconstruction loss:', error,'Discriminator Output:', model_predicts[1][0][0])
    #print("Model Predicts:",model_predicts)

    montage = np.hstack((input_image,reconstructed_image))
    
    if show_images:
        #cv2.imshow("montage",montage)
        #cv2.waitKey()            
        input_image_res = resize_img(input_image)
        reconstructed_image_res = resize_img(reconstructed_image)
        montage2 = np.hstack((input_image_res,reconstructed_image_res))    
        #cv2.imshow("montage",montage2)
        #cv2.waitKey()

    return montage
# Test on first images

montage = test_reconstruction(0)
for i in range(1,10):
    new_img = test_reconstruction(i)
    montage = np.vstack((montage,new_img))
        
cv2.imshow("montage",montage)
cv2.waitKey()

montage_res = resize_img(montage,400)
cv2.imshow("montage",montage_res)
cv2.waitKey()



