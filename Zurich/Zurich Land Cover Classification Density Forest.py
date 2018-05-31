
# coding: utf-8

# # Zurich Land Cover Classification
# 
# This script presents a visualization of training a U-Net classifier on 7 out of 8 available land cover classes of the Zurich dataset, and detecting the unseen class using a Density Forest.

# ## 1. Import Libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# python libraries
from IPython.core.display import Image, display
from matplotlib.patches import Rectangle
import natsort as ns
from multiprocessing import cpu_count
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, FloatSlider
from sklearn.manifold import TSNE
import sys
import pandas as pd

# custom libraries
from helpers.helpers import *
from helpers.data_augment import *
from keras_helpers.unet import *
from keras_helpers.helpers import *
from keras_helpers.callbacks import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sklearn import decomposition
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.models import load_model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# custom libraries
base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code') # Path to density Tree package
sys.path.append(base_dir + '/SIE-Master/Code/density_tree') # Path to density Tree package
from density_tree.density_forest import *
from density_tree.helpers import print_density_tree_latex
from density_tree.plots import plot_tsne, plot_pca
from helpers.helpers import imgs_stretch_eq
from helpers.plots import *


# In[2]:


class_to_remove = 1


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


# In[5]:


# visualize some patches 
imgs_row = 8
fig, axes = plt.subplots(3, imgs_row)
fig.set_size_inches(20, 8)
offset = 0
alpha = .6
for i in range(offset, offset + imgs_row):
    axes[0][i - offset].imshow(x_test[i][..., :3])  # images
    axes[1][i - offset].imshow(
        gt_label_to_color(y_test[i], colors) * alpha + x_test[i][..., :3] * (1 - alpha))  # ground truth (overlay)
    axes[2][i - offset].imshow(gt_label_to_color(y_test[i], colors))  # ground truth

# corresponding part of image
plt.figure(figsize=(10, 5))
plt.imshow(imgs[16][:64, :64 * 8, :3])


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

# In[6]:


# create copies of original data
y_train_label = y_train.copy()
y_val_label = y_val.copy()
y_test_label = y_test.copy()


# In[7]:


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


# In[8]:


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

# In[9]:


# data augmentation
img_idx = 14
im_vis, gt_vis = augment_images_and_gt(x_train[img_idx], y_train_label[img_idx], rf_h=True,
                                                   rf_v=True, rot=True)

fig, axes = plt.subplots(1, 3)
fig.set_size_inches((10, 5))
axes[0].imshow(x_train[img_idx][..., :3])
axes[1].imshow(im_vis[..., :3])
axes[2].imshow(gt_vis)
plt.show()


# In[10]:


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


# In[11]:


# train or load model
# train the model
#model_unet = get_unet(n_classes, x_train.shape[1:])
#model_train(model_unet, data_augmentation=True)
#model_unet.save('models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower() + '_2.h5')  # save model, weights


# In[12]:


# load model
name_model = path + '/models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower() + '.h5'    
model_unet = load_model(name_model, custom_objects={'fn': ignore_background_class_accuracy(0)})


# ### 3.2. Prediction on Test Set

# In[13]:


# get prediction
y_pred = model_unet.predict(x_test, batch_size=20, verbose=1)

# prediction patches without overlapping patches
y_pred = np.concatenate(remove_overlap(imgs, y_pred, ids_test, 64, 32))

# get label
y_pred_label = get_y_pred_labels(y_pred, class_to_remove=class_to_remove)

# get accuracy as softmax pseudo-probability
y_pred_acc = np.max(y_pred, axis=-1)

# Get accuracy as margin between highest and second highest class
y_pred_acc_margin = get_accuracy_probas(y_pred)


# In[14]:


# prediction image
y_pred_acc_imgs = [convert_patches_to_image(imgs, y_pred_acc[..., np.newaxis],
                                       img_idx=idx_im, img_start=ids_test[0], patch_size=64,
                                       stride=64) for idx_im in ids_test]


# In[15]:


# plot prediction results
im_idx = 15
alpha = .3  # for overlay
fig, axes = plt.subplots(1, 6)
fig.set_size_inches(20, 20)
fig_im = x_test[im_idx][..., :3] * (1 - alpha)
fig_test = gt_label_to_color(y_test_label[im_idx], colors)
fig_pred = gt_label_to_color(y_pred_label[im_idx], colors)

# plots
axes[0].imshow(fig_im)
axes[1].imshow(fig_test)
axes[2].imshow(fig_test * alpha + fig_im * (1 - alpha))
axes[3].imshow(fig_pred)
axes[4].imshow(fig_pred * alpha + fig_im * (1 - alpha))
axes[5].imshow(fig_im * 0 + 1)

# titles
axes[0].set_title("Test image")
axes[1].set_title("Ground truth")
axes[2].set_title("Ground truth (overlay)")
axes[3].set_title("Predicted Image")
axes[4].set_title("Predicted Image (overlay)")
axes[5].set_title("Legend")

# legend
legend_data = [[l[0], l[1]] for l in legend.items()]
handles = [Rectangle((0, 0), 1, 1, color=[v / 255 for v in c]) for n, c in legend_data]
labels = np.asarray([n for n, c in legend_data])
axes[5].legend(handles, labels)

# show certitude by network
fig = plt.figure()
plt.imshow(y_pred_acc[im_idx], cmap='gray')
plt.title("Network confidence")
plt.colorbar()


# In[16]:


y_pred_im = [convert_patches_to_image(imgs, gt_label_to_color(y_pred_label, colors), img_idx=i, img_start=16, patch_size=64,
                             stride=64) for i in ids_test]

for img_idx in ids_test:
    # Pred
    plt.figure(figsize=(8,8))
    plt.imshow(y_pred_im[img_idx-16])  # prediction
    plt.axis('off')
    plt.savefig("../Figures/Pred/im_" + str(img_idx+1) + "_pred_wo_cl_" + str(class_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

    # GT
    plt.figure(figsize=(8,8))
    plt.imshow(gt_label_to_color(gt[img_idx],colors))  # gt stitched together
    plt.axis('off')
    plt.savefig("../Figures/Im/gt_" + str(img_idx+1) + ".pdf", bbox_inches='tight', pad_inches=0)


    # show also original image
    plt.figure(figsize=(7, 7))
    plt.imshow(imgs[img_idx][:, :, :3])
    plt.axis('off')
    plt.savefig("../Figures/Im/im_" + str(img_idx+1) + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.title("original image")


# ### 3.3. Accuracy Metrics (Test Set)

# In[17]:


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


# In[18]:


# print to log file
precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test_flattened[filter_items], y_pred_flattened[filter_items])
df = pd.DataFrame(data={'Precision':precision,
                        'Recall':recall,
                       'f1-score':fscore,
                       'support':support}, index=names_keep)

df.index.name = 'Class'
with open("models_out/acc_class_" + str(class_to_remove) + ".csv", 'w') as f:
    print(df.to_latex(float_format='%.3f'), file=f)  # Python 3.x


# ## 4. Certainty using Density Forest
# ### 4.1. Retrieve Activations, PCA, t-SNE

# In[19]:


# image, layer indexes
layer_idx = -2
img_idx = 2
batch_size = 20

# get activations for training Density Forest
act_train = get_activations(imgs, model_unet, layer_idx, x_train, ids_train, batch_size=160, patch_size=64, stride=64)


# In[20]:


# get activations for seen classes

# retain only activation weights for which there is a ground truth
filter_seen = (y_train_label != 0) & (y_train_label != class_to_remove)
act_train_seen = np.concatenate(act_train)[filter_seen] 

# all but those belonging to background
act_train = np.concatenate(act_train)[y_train_label != 0]


# In[21]:


# get activations for testing Density Forest
act_test = get_activations(imgs, model_unet, layer_idx, x_test, ids_test, batch_size=160, patch_size=64, stride=32)

# remove test activations overlap
act_test = remove_overlap(imgs, np.concatenate(act_test), ids_test, patch_size=64, stride=32) 

# all labels, including background
act_test = np.concatenate(act_test, axis=0)[y_test_nostride < np.infty] # convert to 1D


# In[22]:


# get balanced data subset to show in figure
pts_per_class = 100  # same number of points per class
dataset_subset_indices = []
for class_label in range(1, 9):
    ds_subset_ind = np.where(y_test_nostride[y_test_nostride<np.infty]==class_label)[0]
    dataset_subset_indices.append(np.random.choice(ds_subset_ind, size=pts_per_class, replace=False))


# In[23]:


# t-SNE visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_all = tsne.fit_transform(act_test[np.concatenate(dataset_subset_indices)])

plot_tsne(tsne_all, class_to_remove, classes_to_keep, pts_per_class, colors, names)


# In[24]:


# create density tree for activation weights of training data
# PCA
pca = decomposition.PCA(n_components=5)
pca.fit(act_train)  # fit on training set without background pixels
n_components = np.alen(pca.explained_variance_ratio_)
print("Variance explained by first %i components: %.2f" % (
    n_components, sum(pca.explained_variance_ratio_)))

# transform training activations
act_train_seen = pca.transform(act_train_seen)

# transform test set activations
act_test = pca.transform(act_test)


# In[25]:


# t-SNE visualization after PCA
tsne_all = tsne.fit_transform(act_test[np.concatenate(dataset_subset_indices)])

# plot
plot_tsne(tsne_all, class_to_remove, classes_to_keep, pts_per_class, colors, names)


# In[26]:


# plot PCA point
plot_pca(act_test, class_to_remove, classes_to_keep, names, dataset_subset_indices, colors)

print("Variance explained by first 3 components: %.2f" % np.sum(pca.explained_variance_ratio_[:3]))


# ### 4.2. Train Density Forest

# **Parameters importance:**
# - `n_trees`: minor relevance (10 is enough)
# - `min_subset`: important, smaller number <-> more clusters (requires `subset_data` to be high enough)
# - `subset_data`: irrelevant, but should be higher if `min_subset` is lower
# - `max_depth`: important, greater depth <-> more clusters
# - `fact_improvement`: important, smaller minimum factor <-> more clusters
# - `n_max_dim`: unimportant, better to set to 0
# 

# In[42]:


n_trees = 10
max_depth = 3
subsample_pct = .005
min_subset = .001
fact_improvement = .1
n_max_dim = -1
n_jobs=-1

root_nodes_seen = df_create(act_train_seen, max_depth=max_depth, min_subset=min_subset, n_trees=n_trees, 
                                        n_max_dim=n_max_dim, subsample_pct=subsample_pct, n_jobs=n_jobs, verbose=10, 
                                        fact_improvement=fact_improvement)


# In[54]:


# get probabilities for all images
probas = df_traverse_batch(act_test, root_nodes_seen, n_jobs=-1, batch_size=10000, verbosity = 10, standardize=False)


# ### 4.3. Post-Treatment

# In[55]:


# reshape probas to (n_patches, patch_size, patch_size)
patches_start = get_offset(imgs, 64, 64, 16, 16) # idx of first patch in image
patches_end = get_offset(imgs, 64, 64, 16, 20) # idx of first patch in image
n_patches = patches_end - patches_start
probas_seen_im = np.reshape(probas, (n_patches,patch_size,patch_size))

# transformations
#probas_seen_im[probas_seen_im==0]=1e-5  # for log
#probas_seen_im = np.log(probas_seen_im)
probas_seen_im -= np.nanmin(probas_seen_im)
probas_seen_im /= np.nanmax(probas_seen_im)
#probas_seen_im = 1-probas_seen_im

# remove outliers
#probas_seen_im[abs(probas_seen_im - np.nanmean(probas_seen_im)) > m * np.nanstd(probas_seen_im)] = np.nan


# In[56]:


# save probabilities corresponding to an image in an array
probas_imgs = [] # (n_imgs, n_patches, patch_size, patch_size)
for idx_im in ids_test:
    patches_start = get_offset(imgs, 64, 64, 16, idx_im) # idx of first patch in image
    patches_end = get_offset(imgs, 64, 64, 16, idx_im+1) # idx of last patch in image
    probas_im = np.asarray(probas_seen_im[patches_start:patches_end])
    probas_imgs.append(probas_im)


# ### 4.4. Figures, Tables

# In[57]:


for idx_im in ids_test:
    im_cert_out = convert_patches_to_image(imgs, probas_imgs[idx_im-16][..., np.newaxis],
                                           img_idx=idx_im, patch_size=64,
                                           stride=64, img_start=idx_im)

    
    #im_cert_out = imgs_stretch_eq([im_cert_out])
    #im_cert_out = im_cert_out[...,0][0]
    im_cert_out = im_cert_out[...,0]
    
    plt.figure(figsize=(16, 20))
    plt.imshow(im_cert_out, cmap='gray')
    plt.axis('off')
    plt.savefig("../Figures/DF/cl_" +  str(class_to_remove) + "/im_" + str(idx_im + 1) + "_confidence_cl_" + str(class_to_remove) +"_DF.pdf", bbox_inches='tight', pad_inches=0);

    plt.figure(figsize=(16, 20))
    plt.imshow(y_pred_acc_imgs[idx_im-16][...,0], cmap='gray')
    plt.axis('off')
    plt.savefig("../Figures/DF/cl_" +  str(class_to_remove) + "/im_" + str(idx_im + 1) + "_confidence_cl_" + str(class_to_remove) +"_Net.pdf", bbox_inches='tight', pad_inches=0);


# In[58]:


# convert patches to image
idx_im = 16

# show certitude by network
# image, overlay
im_cert_out = convert_patches_to_image(imgs, probas_imgs[idx_im-16][..., np.newaxis],
                                       img_idx=idx_im, patch_size=64,
                                       stride=64, img_start=idx_im)[...,0]

#im_cert_out = imgs_stretch_eq([im_cert_out])[0]

def fig_uncertainty(thresh_2, thresh_3=.3, save=False, show=True):
    
    # part of image overlapping with uncertainty image
    img_part = imgs[idx_im][:im_cert_out.shape[0],:im_cert_out.shape[1],:3]
    fact_mult = 1
    im_overlay = get_fig_overlay_fusion(img_part, im_cert_out*fact_mult, y_pred_acc_imgs[idx_im-16][...,0],
                             thresh_2=thresh_2, thresh_3=thresh_3, opacity=.5)
    
    im_overlay_1 = get_fig_overlay(img_part, im_cert_out*fact_mult,
                             thresh=thresh_2, opacity=.5)
    
    im_overlay_2 = get_fig_overlay(img_part, y_pred_acc_imgs[idx_im-16][...,0],
                                   thresh=thresh_3, opacity=.5)
    # y_pred_acc_imgs[idx_im-16][...,0]

    fig = plt.figure(figsize=(10,12))
    plt.imshow(im_overlay)
    plt.axis('off')
    if save:
        plt.savefig("../Figures/DF/cl_" +  str(class_to_remove) + "/both_thresh_im_" + str(idx_im+1) + "wo_cl_" +
                    str(names[class_to_remove]).lower() + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(10,12))
    axes[0].imshow(im_overlay_1)
    axes[0].set_axis_off()
    
    axes[1].imshow(im_overlay_2)
    axes[1].set_axis_off()

        

min_data, max_data = 0, 1  #  for normal
n_steps = 20
range_1 = (min_data,max_data,(max_data-min_data)/n_steps)

min_data, max_data = .8, 1  #  for net
n_steps = 20
range_2 = (min_data,max_data,(max_data-min_data)/n_steps)
interact(fig_uncertainty, thresh_2=range_1, thresh_3=range_2)

plt.figure(figsize=(10,10))
plt.imshow(gt_label_to_color(gt[idx_im], colors))


# In[59]:


# calculate average certainty by Density Forest
# support for all labels in y_true
support = np.unique(y_test_flattened, return_counts=True)[1][1:]

probas_patches = np.concatenate(probas_imgs) # all patches concatenated (like y_test)
av_cert = []
nans = []
for label in np.arange(1, 9):
    av_cert.append(np.nanmean(probas_patches[y_test_nostride==label]))
    nans.append(np.sum(np.isnan(probas_patches[y_test_nostride==label]))/np.sum(np.ones(np.shape(probas_patches[y_test_nostride==label]))))

av_cert_w = (av_cert*support)/sum(support)
av_cert = np.asarray(av_cert)
nans = np.asarray(nans)

print("Average certainty within class:")
for idx, w in enumerate(av_cert):
    print("%15s: %3.5f, nans: %.2f%%" % (names[idx + 1], w*1e5, nans[idx]*100))


# ratio unseen class / seen classes
cert_unseen = av_cert[class_to_remove - 1]
cert_seen = np.nanmean(np.asarray(av_cert)[av_cert != cert_unseen])

av_cert_w = (av_cert*support)/sum(support)
cert_unseen_w = av_cert_w[class_to_remove - 1]
cert_seen_w = np.nanmean(np.asarray(av_cert)[av_cert_w != cert_unseen])

print("Average certainty unseen class:\t%.5f" % cert_unseen)
print("Average certainty seen classes:\t%.5f" % cert_seen)
print("Ratio between support-weighted cert. of seen classes / unseen class:\t%.3f" % (cert_seen_w / cert_unseen_w))


# In[60]:


# print to log file
df = pd.DataFrame(data={'Average Certainty':av_cert,
                        'Nans':nans}, index=names[1:])

df.index.name = 'Class'
with open("models_out/acc_DF_class_" + str(class_to_remove) + ".csv", 'w') as f:
    print(df.to_latex(float_format='%.3f'), file=f)  # Python 3.x


# In[61]:


# calculate average certainty by Network
av_cert = []
nans = []
for label in np.arange(1, 9):
    av_cert.append(np.nanmean(y_pred_acc[y_test_nostride==label]))
av_cert = np.asarray(av_cert)
nans = np.asarray(nans)

print("Average certainty within class:")
for idx, w in enumerate(av_cert):
    print("%15s: %3.5f" % (names[idx + 1], w))

# ratio unseen class / seen classes
cert_unseen = av_cert[class_to_remove - 1]
cert_seen = np.nanmean(np.asarray(av_cert)[av_cert != cert_unseen])

av_cert_w = (av_cert*support)/sum(support)
cert_unseen_w = av_cert_w[class_to_remove - 1]
cert_seen_w = np.nanmean(np.asarray(av_cert)[av_cert_w != cert_unseen])

print("Average certainty unseen class:\t%.5f" % cert_unseen)
print("Average certainty seen classes:\t%.5f" % cert_seen)
print("Ratio between support-weighted cert. of seen classes / unseen class:\t%.3f" % (cert_seen_w / cert_unseen_w))


# ### 4.5. Accuracy ratios
# 
# | Accuracy indicator | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 
# | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
# | 1A | 4.689 |  | |  | | | | |
# | 1B | 4.980 | 6.792 | 6.038 | 3.489 | 53.949 | 165.043 | 153.863 | 179.750 |
# | 2A | 123.389 | 475.483 | 16.604 | 5.188 | 91.749 | 12332.699 | 43.022 | 79770.728 |
# | 2B | 4.840 | 4.512 | 4.234 | 3.256 | 108.226 | 409.266 | 104.420 | 8511.800 |
# 
# **Abbreviations**
# 1. Network
#   1. `softmax` pseudo-probability ratio weighted
#   2. `max-margin` ratio weighted
# 2. Density Forest
#   1. average weighted accuracy ratios (`DF with randomization`)
#   2. average weighted accuracy ratios (`DF std. with randomization`)
#   
# #### Observations
# - *Higher number of splits* (greater depth, min subset of data per leaf smaller): better ratio, but roads and buildings both very low

# In[62]:


#net_msr = [4.689, 5.581, 5.007,]
net_margin = [4.980 , 6.792, 6.038 , 3.489 , 53.949 , 165.043 , 153.863 , 179.750]
df = [123.389 , 475.483 , 16.604 , 5.188, 91.749 , 12332.699 , 43.022 , 79770.728]
df_std = [4.840 , 4.512 , 4.234 , 3.256 , 108.226 , 409.266 , 104.420 , 8511.800]

cl_idx = 10
#print("%.3f" % (np.dot(net_msr[:cl_idx],support[:cl_idx])/sum(support[:cl_idx])))
print("%.3f" % (np.dot(net_margin[:cl_idx],support[:cl_idx])/sum(support[:cl_idx])))
print("%.3f" % (np.dot(df[:cl_idx],support[:cl_idx])/sum(support[:cl_idx])))
print("%.3f" % (np.dot(df_std[:cl_idx],support[:cl_idx])/sum(support[:cl_idx])))


# In[63]:


# Get precision, recall curve for all pixels in prediction that are not in background
y_scores = y_pred_acc.flatten()
y_true = (y_test_nostride == y_pred_label).flatten()

filt = y_test_nostride.flatten()!=0
y_true = y_true[filt]
y_scores = y_scores[filt]
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
plot_precision_recall(precision, recall, s_name="../Figures/DF/AUC_pred_wo_cl_" + str(class_to_remove) + ".pdf")


# In[64]:


# Get precision, recall curve for all pixels in prediction that are not in background
y_scores = probas_patches.flatten()
y_true = (y_test_nostride == y_pred_label).flatten()

filt = y_test_nostride.flatten()!=0
y_true = y_true[filt]
y_scores = y_scores[filt]
precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
plot_precision_recall(precision, recall, s_name="../Figures/DF/AUC_pred_wo_cl_" + str(class_to_remove) + ".pdf")#y_scores = probas_patches.flatten()

