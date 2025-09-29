import os

import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpts
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.io import loadmat
from tqdm import tqdm
from model_Mixer import MixerSENet
from tensorflow.keras.callbacks import ModelCheckpoint
import scipy.io as sio

X = sio.loadmat('datasets\Houston13_HSI.mat')['data_HS_HR']
Tr_labels = sio.loadmat('datasets\Houston_TrainImage.mat')['TrainImage']
Te_labels = sio.loadmat('datasets\Houston_TestImage.mat')['TestImage']
gt = Tr_labels + Te_labels
NUM_CLASS = Te_labels.max()
class_name = ['Healthy Grass','Stressed Grass','Synthetic Grass',
              'Tree','Soil','Water','Residential','Commercial','Road','Highway',
              'Railway','Parking Lot1','Parking Lot2','Tennis Court','Running Track']

_, labels = get_img_indexes(gt, removeZeroindexes = True)
img_display(classes=Te_labels,title='groundtruth',class_name=class_name)

data = applyPCA(X, numComponents = 15, normalization = True)


# Get class map indexes
X_train_idx, y_train = get_img_indexes(Tr_labels, removeZeroindexes = True)
X_test_idx, y_test = get_img_indexes(Te_labels, removeZeroindexes = True)

num_samples_overall = len(X_train_idx)+len(X_test_idx)

X_train_idx, X_val_idx, y_train, y_val = splitTrainTestSet(X_train_idx, y_train, testRatio = 0.50)
print("Training Percentage =", format(100*len(X_train_idx)/num_samples_overall, ".2f"),'%')
print("Validation Percentage =", format(100*len(X_val_idx)/num_samples_overall, ".2f"),'%')
print("Testing Percentage =", format(100*len(X_test_idx)/num_samples_overall, ".2f"),'%')

sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in np.unique(gt):
    if i == 0: continue
    sample_report += f"{class_name[i-1]: ^25}{(y_train==i-1).sum(): ^10}{(y_val==i-1).sum(): ^10}{(y_test==i-1).sum(): ^10}{(gt==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(y_train-1): ^10}{len(y_val): ^10}{len(y_test): ^10}{len(labels): ^10}"
print(sample_report)


window_size = 9
X_train = createImageCubes(data, X_train_idx, window_size)
y_train = keras.utils.to_categorical(y_train)

X_val = createImageCubes(data, X_val_idx, window_size)
y_val = keras.utils.to_categorical(y_val)

depth = 5
model = MixerSENet(img_list = X_train, NumClasses = NUM_CLASS, depth = depth, filters = 64)
model.summary()

from net_flops import net_flops

net_flops(model)



  
checkpoint = ModelCheckpoint(
    f"Houston_MixerSE.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
# Define a callback to modify the learning rate dynamically
lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=10,
    min_lr=5e-5
    )


history = model.fit(X_train, y_train,
                    epochs = 100,
                    batch_size = 64,
                    validation_data = (X_val, y_val),
                    callbacks=[checkpoint, lr_callback],
                    )

display_history(history)
    
model.load_weights(f"Houston_MixerSE.h5")
    
Y_pred_test = predict_by_batching(model, input_tensor_idx = X_test_idx, batch_size = 1000, X = data, windowSize = window_size)
y_pred_test = np.argmax(Y_pred_test, axis=1)
    
kappa = cohen_kappa_score(y_test,  y_pred_test)
oa = accuracy_score(y_test, y_pred_test)
cm = confusion_matrix(y_test, y_pred_test)
class_acc = cm.diagonal() / cm.sum(axis=1)
aa = np.mean(class_acc)
    
print("Overall Accuracy = ", float(format((oa)*100, ".2f"))) 
print("Average Accuracy = ", float(format((aa)*100, ".2f")))
print('Kappa = ', float(format((kappa)*100, ".2f")))
 


model.load_weights(f"Houston_MixerSE.h5")

Predicted_Class_Map = get_class_map(model, data, gt, window_size)
img_display(classes=Predicted_Class_Map, title='Predicted', class_name=class_name)

gt_binary = gt.copy()
gt_binary[gt>0]=1
img_display(classes=Predicted_Class_Map*gt_binary, title='Predicted with Mask', class_name=class_name)


Folder = 'Matlab_Outputs/'
Name = f'Houston_MixerSENet'
sio.savemat(Folder + Name+'.mat', {Name: Predicted_Class_Map})



