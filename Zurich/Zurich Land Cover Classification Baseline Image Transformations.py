
# coding: utf-8

# # Zurich Land Cover Classification
# 
# This script presents a visualization of training a U-Net classifier on 7 out of 8 available land cover classes of the Zurich dataset, and detecting the unseen class using the following Baseline Method:
# ## Confidence from Invariance to Image Transformations
# https://arxiv.org/pdf/1804.00657.pdf
# 
# Data Visualizations are contained in the notebook `Zurich Land Cover Density Forest.ipynb`

# ## 1. Import Libraries

# In[1]:


# python libraries
from multiprocessing import cpu_count
from sklearn import metrics
import sys

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.models import load_model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Meta-Parameters
#base_dir = '/Users/cyrilwendl/Documents/EPFL'  # for local machine
base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code') # Path to density Tree package Tree package

# custom libraries
from helpers.helpers import *
from helpers.data_augment import *
from helpers.plots import *
from keras_helpers.unet import *
from keras_helpers.helpers import *
from keras_helpers.callbacks import *
from baselines.helpers import *
from baselines.plots import *


# In[2]:
plt.ioff()

class_to_remove = int(sys.argv[1])
print(class_to_remove)


# ## 2. Load Images

# In[3]:


path = os.getcwd()

imgs, gt = load_data(path)

# gt to labels
# Next, we need to convert the ground truth (colors) to labels 
legend = OrderedDict((('Background', [255, 255, 255]),
                      ('Roads', [0, 0, 0]),
                      ('Buildings', [100, 100, 100]),
                      ('Trees', [0, 125, 0]),
                      ('Grass', [0, 255, 0]),
                      ('Bare Soil', [150, 80, 0]),
                      ('Water', [0, 0, 150]),
                      ('Railways', [255, 255, 0]),
                      ('Swimming Pools', [150, 150, 255])))

# get class names by increasing value (as done above)
names, colors = [], []
for name, color in legend.items():
    names.append(name)
    colors.append(color)

gt = gt_color_to_label(gt, colors)


# ### 2.2. Get patches

# In[4]:


# Get patches
patch_size = 64
stride_train = 64  # has to be <= patch_size
stride_test = 32  # has to be <= patch_size

# ids for training, validation and test sets (0-19)
ids_train = np.arange(0, 12)
ids_val = np.arange(12, 16)
ids_test = np.arange(16, 20)

# get training, test and validation sets
x_train = get_padded_patches(imgs[ids_train], patch_size=patch_size, stride=stride_train)
x_val = get_padded_patches(imgs[ids_val], patch_size=patch_size, stride=stride_train)
x_test = get_padded_patches(imgs[ids_test], patch_size=patch_size, stride=stride_test)
x_test_nostride = get_padded_patches(imgs[ids_test], patch_size=patch_size, stride=patch_size)

y_train = get_gt_patches(gt[ids_train], patch_size=patch_size, stride=stride_train)
y_val = get_gt_patches(gt[ids_val], patch_size=patch_size, stride=stride_train)
y_test = get_gt_patches(gt[ids_test], patch_size=patch_size, stride=stride_test)
y_test_nostride = get_gt_patches(gt[ids_test], patch_size=patch_size, stride=patch_size)

print(x_test.shape)
print(x_test_nostride.shape)
print(y_test_nostride.shape)


# ## 3. Keras CNN
# 
# Data Split: 
# - Training: 12 images
# - Validation: 4 images
# - Test: 4 images
# 
# Tested Architectures: 
# 
# | Model | Patch Size | Data Augmentations | Number of Parameters | Testing Precision (avg) | Testing Recall (avg) | Testing f1 score (avg) | Validation / Test accuracy |
# | ------- | ------- | ------- | ------- | ------- | ------- |
# | U-Net | 64 | Rot 90°, Flipping  | 7,828,200 | 0.87 | 0.858 | 0.86 | t |
# | U-Net | 128 | Rot 90°, Flipping  | 7,828,200 | 0.69 | 0.61 | 0.64 | t |
# | U-Net | 128 | Rot 90°, Flipping  | 7,828,200 | 0.90 | 0.89 | 0.89 | v |

# In[5]:


# create copies of original data
y_train_label = y_train.copy()
y_val_label = y_val.copy()
y_test_label = y_test.copy()


# In[6]:


# get class weights
labels_unique = np.unique(y_train.flatten())
print(labels_unique)
class_weights = class_weight.compute_class_weight('balanced', labels_unique, y_train.flatten())
class_weights[0] = 0  # give less weight to background label class
class_weights[5] = 7  # give less weight to bare soil class
class_weights[8] = 7  # give less weight to swimming pool class

print("Class weights:")
for i, w in enumerate(class_weights):
    print("%15s: %3.3f" % (names[i], w))


# In[7]:


n_classes = 9

# convert to numpy arrays
x_train = np.asarray(x_train)
x_val = np.asarray(x_val)
x_test = np.asarray(x_test)

# make y data categorical
y_train = to_categorical(y_train_label, n_classes)
y_val = to_categorical(y_val_label, n_classes)

# remove class
classes_to_keep = np.asarray([x for x in range(1, n_classes) if x != class_to_remove])

names_keep = np.asarray(names)[classes_to_keep]
names_keep = names_keep.tolist()
print("classes to keep: " + str(names_keep))

y_train = y_train[..., classes_to_keep]
y_val = y_val[..., classes_to_keep]
n_classes = len(classes_to_keep)
class_weights = class_weights[classes_to_keep]

# print shapes of variables
for var in x_train, y_train, x_val, y_val:
    print(np.shape(var))


# ### 3.1. Train CNN

# In[8]:


# callbacks (evaluated every epoch)
# show loss and accuracy figures after each epoch
callback_plot = PlotLosses()

# stop early if after several epochs the accuracy doesn't improve
callback_earlystop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=24, verbose=1, mode='auto')

# decrease learning rate when accuracy stops improving
callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, verbose=1, mode='auto',
                                epsilon=1e-4, cooldown=0, min_lr=1e-8)

# checkpoint to save weights at every epoch (in case of interruption)
file_path = "weights-improvement.hdf5"
callback_checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

callback_tensorboard = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=True)

# model setup
batch_size = 20
epochs = 300


def model_train(model, data_augmentation):
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(batch_generator(x_train, y_train,
                                        batch_size=batch_size, data_augmentation=data_augmentation),
                        steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
                        epochs=epochs,
                        verbose=1,
                        class_weight=class_weights,  # weights for loss function
                        validation_data=(x_val, y_val),
                        callbacks=[callback_earlystop,
                                   callback_lr,
                                   #callback_checkpoint,
                                   callback_plot,
                                   callback_tensorboard],
                        workers=cpu_count(),
                        use_multiprocessing=True)


# In[9]:


# train or load model
# train the model
#model_unet = get_unet(n_classes, x_train.shape[1:])
#model_train(model_unet, data_augmentation=True)
#model_unet.save('models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower() + '_2.h5')  # save model, weights

# load model
name_model = path + '/models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower() + '.h5'    
model_unet = load_model(name_model, custom_objects={'fn': ignore_background_class_accuracy(0)})


# ### 3.2. Prediction on Test Set

# In[10]:


# get prediction
y_pred = model_unet.predict(x_test, batch_size=20, verbose=1)

# prediction patches without overlapping patches
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))

# get label
y_pred_label = get_y_pred_labels(y_pred, class_to_remove=class_to_remove)

# Get accuracy as margin between highest and second highest class
y_pred_acc = get_acc_net_max_margin(y_pred)


# In[11]:


# prediction image
y_pred_acc_imgs = [convert_patches_to_image(imgs, y_pred_acc[...,np.newaxis],
                                       img_idx=idx_im, img_start=ids_test[0], patch_size=64,
                                       stride=64) for idx_im in ids_test]


# ### 3.3. Accuracy Metrics (Test Set)

# In[12]:


# Accuracy metrics
y_pred_flattened= np.asarray(y_pred_label.flatten()).astype('int')
y_test_flattened= np.asarray(y_test_nostride.flatten()).astype('int')

# mask background and removed classes for evaluation metrics
filter_items = (y_test_flattened != 0) & (y_test_flattened != class_to_remove)

# Class accuracy, average accuracy
print(metrics.classification_report(
    y_test_flattened[filter_items],
    y_pred_flattened[filter_items],
    target_names=names_keep,
    digits=3))


# Overall accuracy
OA = metrics.accuracy_score(y_test_flattened[filter_items], y_pred_flattened[filter_items])
print("Overall accuracy: %.3f %%" % (OA*100))


# ## Confidence from Invariance to Image Transformations
# https://arxiv.org/pdf/1804.00657.pdf

# In[13]:


# get prediction
y_pred = model_unet.predict(x_test, batch_size=20, verbose=1)
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))

# prediction patches without overlapping patches
# y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))

# get label
y_pred_label = get_y_pred_labels(y_pred, class_to_remove=class_to_remove)

y = np.equal(y_pred_label, y_test_nostride)


# In[14]:


x_aug, y_aug = augment_images_and_gt(x_test_nostride, x_test_nostride, gamma=.8, force=True)

# visualize an image
im_idx = 5
fig, axes  = plt.subplots(1,2, figsize=(8,4))
axes[0].imshow(x_test_nostride[im_idx][...,:3])
axes[1].imshow(x_aug[im_idx][...,:3])


# In[15]:


y_preds = []
# original
y_pred = model_unet.predict(x_test, verbose=1)
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))
y_preds.append(y_pred)

# TODO horizontal flipping
#x_aug = [np.fliplr(x_test[i]) for i in range(len(x_test))]
#y_preds.append(model.predict(x_aug))
#x_aug, _ = augment_images_and_gt(x_test, x_test, rf_h=True)
#y_preds.append(model_unet.predict(x_aug, verbose=1))

# gamma
x_aug, _ = augment_images_and_gt(x_test, x_test, gamma=.2, force=True)
fig, axes  = plt.subplots(1,2, figsize=(8,4))
axes[0].imshow(x_test[im_idx][...,:3])
axes[1].imshow(x_aug[im_idx][...,:3])

y_pred = model_unet.predict(x_aug, verbose=1)
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))
y_preds.append(y_pred)

# contrast
x_aug, _ = augment_images_and_gt(x_test, x_test, contrast=1.5, force=True)
fig, axes  = plt.subplots(1,2, figsize=(8,4))
axes[0].imshow(x_test[im_idx][...,:3])
axes[1].imshow(x_aug[im_idx][...,:3])

y_pred = model_unet.predict(x_aug, verbose=1)
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))
y_preds.append(y_pred)

# contrast
x_aug, _ = augment_images_and_gt(x_test, x_test, brightness=1.2, force=True)
fig, axes  = plt.subplots(1,2, figsize=(8,4))
axes[0].imshow(x_test[im_idx][...,:3])
axes[1].imshow(x_aug[im_idx][...,:3])

y_pred = model_unet.predict(x_aug, verbose=1)
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))
y_preds.append(y_pred)
#x_aug, _ = augment_images_and_gt(x_test, x_test, blur=True)


# In[16]:


y_preds = np.asarray(y_preds)
y_preds = np.transpose(y_preds,(1,2,3,4,0))
y_preds.shape


# In[17]:


idx_train = np.arange(get_offset(imgs,64,64,16,18))
idx_test = np.arange(get_offset(imgs,64,64,16,18),get_offset(imgs,64,64,16,20))

x_train = y_preds[idx_train]
x_test = y_preds[idx_test]

y_train = y[idx_train]
y_test = y[idx_test]

x_train = np.transpose(np.concatenate(np.concatenate((x_train))),(2,0,1))
x_test = np.transpose(np.concatenate(np.concatenate((x_test))),(2,0,1))
y_train = y_train.flatten()
y_test = y_test.flatten()


# In[18]:


t = np.transpose(np.concatenate(np.concatenate(y_preds)),(2,0,1))
y_true = y_test_nostride.flatten()
ind_nobckg = ((np.where(y_true!=0))*1)[0]


# In[19]:


#show_softmax(ind_nobckg[2], t, y_true)


# In[20]:


# MLP, similar to baseline 1

# transformation seem to deteriorate performance
_, x_train = reorder_truncate_concatenate(x_train, n_components=20)
_, x_test = reorder_truncate_concatenate(x_test, n_components=20)
#x_train  = np.concatenate(x_train, axis=-1)
#x_test  = np.concatenate(x_test, axis=-1)


# In[21]:


# convert class vectors to binary class matrices
num_classes = 2

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(np.shape(y_train),np.shape(y_test))
np.shape(x_train),np.shape(x_test)


# In[22]:


# Train MLP
batch_size = 2000
epochs = 20

train = False
if train:
    model_mlp = get_mlp(num_classes, x_train_mlp.shape[1:], 300)
    history = model_mlp.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
    model_mlp.save('models_out/model_MLP_wo_cl_' + str(names[class_to_remove]).lower() + '_2.h5')  # save model, weights


# In[23]:


model_mlp = load_model('models_out/model_MLP_wo_cl_' + str(names[class_to_remove]).lower() + '_2.h5')


# In[24]:


y_pred = model_mlp.predict(x_test, verbose = 1, batch_size=batch_size)
y_pred_label = y_pred[:,1]  # probability to have a good label


# In[25]:


# get patches from pixels
n_patches = get_offset(imgs, 64, 64, 18, 20)
y_pred_label_patches = np.reshape(y_pred_label,(n_patches,64, 64))

# get image from patches
img_idx = 19
y_pred_label_imgs = convert_patches_to_image(imgs, y_pred_label_patches[..., np.newaxis], img_idx, 64, 64, 18)
plt.figure(figsize=(10,10))
plt.imshow(y_pred_label_imgs[...,0])
plt.savefig("../Figures/baseline/im_" + str(img_idx+1) + "_pred_wo_cl_" + str(class_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)


# In[26]:


# calculate average certainty by  MLP
av_cert = np.asarray([np.nanmean(y_pred_label_patches
                      [y_test_nostride[idx_test]==label]) 
           for label in np.arange(1, 9)])
av_cert[-1] = 0


plot_probas(av_cert, class_to_remove, names[1:])
plt.savefig("../Figures/baseline/BL1_probas_" + str(class_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)
# ratio unseen class / seen classes
cert_unseen = av_cert[class_to_remove - 1]
cert_seen = np.nanmean(np.asarray(av_cert)[av_cert != cert_unseen])

# weighted accuracies
# get support (for weighting)
_, support = np.unique(y_test_nostride, return_counts=True)
support = support[1:]
av_cert_w = (av_cert*support)/sum(support)
cert_unseen_w = av_cert_w[class_to_remove - 1]
cert_seen_w = np.nanmean(np.asarray(av_cert)[av_cert_w != cert_unseen])

print("Average certainty unseen class:\t%.5f" % cert_unseen)
print("Average certainty seen classes:\t%.5f" % cert_seen)
print("Ratio between support-weighted cert. of seen classes / unseen class:\t%.3f" % (cert_seen_w / cert_unseen_w))


# ### Recall-Precision Curve

# In[27]:


# MSR
y_scores = y_pred_label.flatten()
y_true = y_test[:,1]
plt.figure()
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
plot_precision_recall(precision, recall, s_name="../Figures/DF/AUC/AUC_pred_wo_cl_" + str(class_to_remove) + "_ImgT.pdf")

