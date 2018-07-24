
# coding: utf-8

# # Zurich Land Cover Classification
# 
# This script presents a visualization of training a U-Net classifier on 7 out of 8 available land cover classes of the Zurich dataset, and detecting the unseen class using a Density Forest.

# ## Import Libraries

# In[1]:


# python libraries
import os, sys
from multiprocessing import cpu_count
from sklearn.manifold import TSNE
import pandas as pd
from sklearn import decomposition, svm, preprocessing
from sklearn.utils import class_weight
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow.python.client import device_lib

# custom libraries
# base_dir = '/Users/cyrilwendl/Documents/EPFL'
base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code')  # Path to density Tree package

from helpers.helpers import *
from helpers.plots import *
from helpers.data_augment import *
from helpers.data_loader import *
from helpers.parameter_search import *
from density_forest.density_forest import *
from density_forest.helpers import *
from baselines.helpers import *
from keras_helpers.unet import *
from keras_helpers.callbacks import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
print(device_lib.list_local_devices())

class_to_remove = int(sys.argv[1])
paramsearch = False  # search for best hyperparameters

my_dpi=255 # dpi of my screen, for image exporting

# data frame with previously found optimal hyperparameters
df_ps = pd.read_csv('models_out/hyperparams.csv', index_col=0) 

# In[2]:


df_ps

# # Load Data

# In[3]:


path = os.getcwd()

# data without overlap
print("loading data")
data_train = ZurichLoader(path, 'train', class_to_remove=class_to_remove)
data_val = ZurichLoader(path, 'val', class_to_remove=class_to_remove)
data_test = ZurichLoader(path, 'test', class_to_remove=class_to_remove)

print("loading data with overlap")
# data with overlap, for prediction
data_train_overlap = ZurichLoader(path, 'train', stride=32, inherit_loader=data_train)
data_val_overlap = ZurichLoader(path, 'val', stride=32, inherit_loader=data_val)
data_test_overlap = ZurichLoader(path, 'test', stride=32, inherit_loader=data_test)

# class names and colors
names = data_train.names
colors = data_train.colors
n_classes = 8
classes_to_keep = np.asarray([x for x in range(1, n_classes) if x != class_to_remove])
names_keep = np.asarray(names)[classes_to_keep]
print("classes to keep: " + str(names_keep))

# In[4]:


for dataset, offset in zip([data_train, data_val, data_test], [0, 10, 15]):
    for im_idx, im in enumerate(dataset.imgs):
        im = im[..., :3]
        f_name = "../Figures/Zurich/Im/Im_" + str(im_idx + offset) + ".jpg"
        export_figure_matplotlib(im, f_name, dpi=my_dpi)

# In[5]:


for dataset, offset in zip([data_train, data_val, data_test], [0, 10, 15]):
    for gt_idx, gt in enumerate(dataset.gt):
        gt_col = gt_label_to_color(gt, colors)*255
        f_name = "../Figures/Zurich/Im/GT_" + str(gt_idx + offset) + ".jpg"
        export_figure_matplotlib(gt_col, f_name, dpi=my_dpi)

# In[6]:


pred_labels_tr, cnt_tr = np.unique(data_train.gt_patches.astype('int'), return_counts=True)
pred_labels_val, cnt_val = np.unique(data_val.gt_patches.astype('int'), return_counts=True)
pred_labels_te, cnt_te = np.unique(data_test.gt_patches.astype('int'), return_counts=True)

cnt_tr = cnt_tr / np.sum(cnt_tr) * 100
cnt_val = np.concatenate((cnt_val / np.sum(cnt_val) * 100, [0]))
cnt_te = cnt_te / np.sum(cnt_te) * 100

df = pd.DataFrame({'Train': cnt_tr, 'Val': cnt_val, 'Test': cnt_te}, index=names[pred_labels_tr])

axis = df[::-1].plot.barh(figsize=(7, 6), colormap='winter')
plt.xlim([0, 100])
plt.xlabel("Count [%]")
plt.grid(alpha=.3)
axis.spines['right'].set_visible(False)
axis.spines['top'].set_visible(False)
plt.savefig("../Figures/Zurich/Pred_count/ZH_dist.pdf", bbox_inches='tight', pad_inches=0)

# # CNN Training
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

# In[7]:


"""
# create copies of original data
y_train_label = y_train.copy()
y_val_label = y_val.copy()
y_test_label = y_test.copy()

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
"""

# In[8]:


"""

# convert to numpy arrays
x_train = np.asarray(x_train)
x_val = np.asarray(x_val)
x_test = np.asarray(x_test)

# make y data categorical
y_train = to_categorical(y_train_label, n_classes)
y_val = to_categorical(y_val_label, n_classes)

y_train = y_train[..., classes_to_keep]
y_val = y_val[..., classes_to_keep]
n_classes = len(classes_to_keep)
class_weights = class_weights[classes_to_keep]

# print shapes of variables
for var in x_train, y_train, x_val, y_val:
    print(np.shape(var))
"""

# ### Train CNN

# In[9]:


"""
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
                                   # callback_checkpoint,
                                   callback_plot,
                                   callback_tensorboard],
                        workers=cpu_count(),
                        use_multiprocessing=True)
                        
"""

# In[10]:


# train or load model
# train the model
# model_unet = get_unet(n_classes, x_train.shape[1:])
# model_train(model_unet, data_augmentation=True)
# model_unet.save('models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower() + '_2.h5')  # save model, weights

# In[11]:


# load model
name_model = path + '/models_out/model_unet_64_flip_rot90_wo_cl_' + str(names[class_to_remove]).lower().replace(" ", "") + '.h5'    
model_unet = load_model(name_model, custom_objects={'fn': ignore_background_class_accuracy(0)})

# ### Predictions

# In[12]:


# get all predictions in training and test set
# training set
y_pred_tr = model_unet.predict(data_train_overlap.im_patches, verbose=1)
y_pred_tr = remove_overlap(data_train.imgs, y_pred_tr, 64, 32)
y_pred_label_tr = get_y_pred_labels(y_pred_tr, class_to_remove=class_to_remove, background=True)

# validation set
y_pred_val = model_unet.predict(data_val_overlap.im_patches, verbose=1)
y_pred_val = remove_overlap(data_val.imgs, y_pred_val, 64, 32)
y_pred_label_val = get_y_pred_labels(y_pred_val, class_to_remove=class_to_remove, background=True)

# test set
y_pred_te = model_unet.predict(data_test_overlap.im_patches, verbose=1)
y_pred_te = remove_overlap(data_test.imgs, y_pred_te, 64, 32)
y_pred_label_te = get_y_pred_labels(y_pred_te, class_to_remove=class_to_remove, background=True)

# In[13]:


# get indices of correctly / incorrectly predicted pixels
# train
#pred_t_tr = (data_train.gt_patches != class_to_remove) & (data_train.gt_patches != 0)
pred_t_tr = y_pred_label_tr == data_train.gt_patches
pred_f_tr = data_train.gt_patches == class_to_remove

# val
#pred_t_val = (data_val.gt_patches != class_to_remove) & (data_val.gt_patches != 0)
pred_t_val = y_pred_label_val == data_val.gt_patches
pred_f_val = data_val.gt_patches == class_to_remove

# test
#pred_t_te = (data_test.gt_patches != class_to_remove) & (data_test.gt_patches != 0)
pred_t_te = y_pred_label_te == data_test.gt_patches
pred_f_te = data_test.gt_patches == class_to_remove

# In[14]:


# export predicted images
offset = 0
for dataset, preds in zip([data_train, data_val, data_test], [y_pred_label_tr, y_pred_label_val, y_pred_label_te]):
    imgs = convert_patches_to_image(dataset.imgs, preds[..., np.newaxis], 64, 64)
    for im_idx, im in enumerate(imgs):
        im_color = gt_label_to_color(im, colors) * 255
        f_name = "../Figures/Zurich/Im_pred/cl_" + str(class_to_remove)
        f_name = f_name + "/Im_" + str(im_idx + offset) + "_wo_cl_" + str(class_to_remove) + ".jpg"
        export_figure_matplotlib(im_color, f_name, dpi=my_dpi)
        
    offset += len(dataset.imgs)

# ### Accuracy Metrics (Test Set)

# In[15]:


# Get oa, aa for train, val, test
# train
y_pred_tr_flattened = np.asarray(y_pred_label_tr.flatten()).astype('int')
y_tr_flattened = np.asarray(data_train.gt_patches.flatten()).astype('int')
filter_items = (y_tr_flattened != 0) & (y_tr_flattened != class_to_remove)
oa_tr = oa(y_tr_flattened[filter_items], y_pred_tr_flattened[filter_items])
aa_tr, aa_tr_cl = aa(y_tr_flattened[filter_items], y_pred_tr_flattened[filter_items])

# val
y_pred_val_flattened = np.asarray(y_pred_label_val.flatten()).astype('int')
y_val_flattened = np.asarray(data_val.gt_patches.flatten()).astype('int')
filter_items = (y_val_flattened != 0) & (y_val_flattened != class_to_remove)
oa_val = oa(y_val_flattened[filter_items], y_pred_val_flattened[filter_items])
aa_val, aa_val_cl = aa(y_val_flattened[filter_items], y_pred_val_flattened[filter_items])

# test
y_pred_te_flattened = np.asarray(y_pred_label_te.flatten()).astype('int')
y_te_flattened = np.asarray(data_test.gt_patches.flatten()).astype('int')
filter_items = (y_te_flattened != 0) & (y_te_flattened != class_to_remove)
oa_te = oa(y_te_flattened[filter_items], y_pred_te_flattened[filter_items])
aa_te, aa_te_cl = aa(y_te_flattened[filter_items], y_pred_te_flattened[filter_items])

print(np.round(np.multiply([oa_tr, aa_tr], 100), 2))
print(np.round(np.multiply([oa_val, aa_val], 100), 2))
print(np.round(np.multiply([oa_te, aa_te], 100), 2))

# In[16]:


# write metrics to CSV files
df_metrics = pd.read_csv('models_out/metrics_ND.csv', index_col=0)
df2 = pd.DataFrame({str(names[class_to_remove]):[oa_tr, aa_tr, oa_val, aa_val, oa_te, aa_te]},
                    index = ['OA Train', 'AA Train', 'OA Val', 'AA Val', 'OA Test', 'AA Test']).T
df_metrics = df_metrics.append(df2)
df_metrics = df_metrics[~df_metrics.index.duplicated(keep='last')]  # avoid duplicates
df_metrics.to_csv('models_out/metrics_ND.csv')
# print((df_metrics*100).round(2).to_latex())

# In[17]:


# Accuracy metrics
y_pred_flattened = np.asarray(y_pred_label_te.flatten()).astype('int')
y_test_flattened = np.asarray(data_test.gt_patches.flatten()).astype('int')

# mask background and removed classes for evaluation metrics
filter_items = (y_test_flattened != 0) & (y_test_flattened != class_to_remove)

# Class accuracy, average accuracy
print(metrics.classification_report(
    y_test_flattened[filter_items],
    y_pred_flattened[filter_items],
    target_names=names_keep,
    digits=3))


# Overall accuracy
print(np.round(oa_te * 100, 2))

# ## Distribution of predictions in unseen class

# In[18]:


# distribution of predicted label
pred_labels, pred_counts = np.unique(y_pred_label_te[pred_f_te], return_counts=True)
pred_counts = pred_counts / sum(pred_counts) * 100

# visualization
fig = plt.figure(figsize=(7, 5))
plt.bar(pred_labels, pred_counts)
plt.xticks(np.arange(0, 10))
plt.ylim([0,100])
plt.xlabel("Predicted Label")
plt.ylabel("Count [%]")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.title("Misclassified labels (mean MSR=%.2f)" % np.mean(get_acc_net_msr(y_pred_te[pred_f_te])))
plt.xticks(pred_labels_te, names, rotation=20)
plt.savefig("../Figures/Zurich/Pred_count/ZH_pred-count_wo_cl" + str(class_to_remove) + ".pdf",
            bbox_inches='tight', pad_inches=0)

# # Novelty Detection

# ## Network

# In[19]:


probas_msr = get_acc_net_msr(y_pred_te).flatten()
probas_margin = get_acc_net_max_margin(y_pred_te).flatten()
probas_entropy = get_acc_net_entropy(y_pred_te).flatten()

# In[20]:


# precision-recall curves

# msr
y_scores = -probas_msr
y_true = pred_f_te.flatten()
precision_msr, recall_msr, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_msr = metrics.average_precision_score(y_true, y_scores)
auroc_msr = metrics.roc_auc_score(y_true, y_scores)
fpr_msr, tpr_msr, _ = metrics.roc_curve(y_true, y_scores)

# margin
y_scores = -probas_margin
precision_margin, recall_margin, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_margin = metrics.average_precision_score(y_true, y_scores)
auroc_margin = metrics.roc_auc_score(y_true, y_scores)
fpr_margin, tpr_margin, _ = metrics.roc_curve(y_true, y_scores)

# entropy
y_scores = -probas_entropy
precision_entropy, recall_entropy, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_entropy = metrics.average_precision_score(y_true, y_scores)
auroc_entropy = metrics.roc_auc_score(y_true, y_scores)
fpr_entropy, tpr_entropy, _ = metrics.roc_curve(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_msr, pr_auc_msr))
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_margin, pr_auc_margin))
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_entropy, pr_auc_entropy))

# In[21]:


# visualization
# MSR
probas_patches_msr = np.reshape(probas_msr, np.shape(data_test.gt_patches))
probas_patches_msr -= np.min(probas_patches_msr)
probas_patches_msr /= np.max(probas_patches_msr)

# margin
probas_patches_margin = np.reshape(probas_margin, np.shape(data_test.gt_patches))
probas_patches_margin -= np.min(probas_patches_margin)
probas_patches_margin /= np.max(probas_patches_margin)

# entropy
probas_patches_entropy = np.reshape(probas_entropy, np.shape(data_test.gt_patches))
probas_patches_entropy -= np.min(probas_patches_entropy)
probas_patches_entropy /= np.max(probas_patches_entropy)

base_folder = "../Figures/Zurich/Im_cert/cl_" + str(class_to_remove)

# In[22]:


acc_im_msr = convert_patches_to_image(data_test.imgs, probas_patches_msr[..., np.newaxis], 64, 64)
acc_im_margin = convert_patches_to_image(data_test.imgs, probas_patches_margin[..., np.newaxis], 64, 64)
acc_im_entropy = convert_patches_to_image(data_test.imgs, probas_patches_entropy[..., np.newaxis], 64, 64)

# export images
for img_idx in range(len(data_test.imgs)):
    # msr
    acc_im_msr_ = exposure.equalize_hist(acc_im_msr[img_idx])
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_msr_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_msr_, f_name, dpi=my_dpi)
    
    # margin
    
    acc_im_margin_ = exposure.equalize_hist(acc_im_margin[img_idx])
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_margin_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_margin_, f_name, dpi=my_dpi)
    
    # entropy
    
    acc_im_entropy_ = exposure.equalize_hist(acc_im_entropy[img_idx])
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_entropy_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_entropy_, f_name, dpi=my_dpi)

# In[23]:


# export colorbar
a = np.array([[0,1]])
plt.figure(figsize=(9, 1.5))
img = plt.imshow(a, cmap="RdYlGn")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.1, 0.4, 0.5])
cb = plt.colorbar(orientation="horizontal", cax=cax)
cb.outline.set_linewidth(0)
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("../Figures/Zurich/Im_cert/colorbar.pdf", bbox_inches='tight', pad_inches=0)

# ## Dropout

# In[24]:


def predict_with_dropout_imgs(model, dataset, batch_size=300, n_iter=10):
    """
    make all predictions per batch for image data
    :param model: model to use for predictions
    :param x: image patches
    :param imgs: original entire images
    :param batch_size: size of prediction batch
    :param n_iter: number of predictions to make with dropout
    :param ids: ids of images for which to make predictions
    :return:
    """
    x = dataset.im_patches
    imgs = dataset.imgs
    n_steps = int(np.ceil(len(x) / batch_size))
    preds_it = []
    f = k.function([model.layers[0].input, k.learning_phase()], [model.layers[-1].output])

    for _ in tqdm(range(n_iter)):
        preds = []
        for i in range(n_steps):
            idx_start = i * batch_size
            idx_end = (i + 1) * batch_size
            pred = np.concatenate(f([x[idx_start:idx_end], 1]))
            preds.append(pred)

        preds = np.concatenate(preds)
        preds = remove_overlap(dataset.imgs, preds, dataset.patch_size, dataset.stride)
        preds_it.append(preds)

    return preds_it


# In[25]:


# get predictions
y_preds = predict_with_dropout_imgs(model_unet, data_test_overlap, batch_size=500, n_iter=10)

# In[26]:


# get prediction and confidence
prediction = np.mean(y_preds, 0)
probas_dropout = -get_acc_net_entropy(prediction)
del y_preds # free memory

# In[27]:


# dropout metrics
y_scores = probas_dropout.flatten()
precision_dropout, recall_dropout, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_dropout = metrics.average_precision_score(y_true, y_scores)
auroc_dropout = metrics.roc_auc_score(y_true, y_scores)
fpr_dropout, tpr_dropout, _ = metrics.roc_curve(y_true, y_scores)
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_dropout, pr_auc_dropout))

# In[28]:


# visualization
probas_patches_dropout = np.reshape(probas_dropout, np.shape(data_test.gt_patches))
probas_patches_dropout -= np.min(probas_patches_dropout)
probas_patches_dropout /= np.max(probas_patches_dropout)

# show image of DF uncertainty vs. max margin uncertainty
acc_im_dropout = convert_patches_to_image(data_test.imgs, -probas_patches_dropout[..., np.newaxis], 64, 64)
for img_idx in range(len(data_test.imgs)):
    acc_im_dropout_ = exposure.equalize_hist(acc_im_dropout[img_idx])
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_dropout_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_dropout_, f_name, dpi=my_dpi)

# ## Retrieve Activations, PCA, t-SNE

# In[91]:


# get activations for training Density Forest
act_train_all = get_activations_batch(model_unet, -2, data_train_overlap.im_patches, 20, verbose=True)

# retain only activation weights for which there is a ground truth
act_train_all = remove_overlap(data_train.imgs, act_train_all, patch_size=64, stride=32)
act_train = act_train_all[pred_t_tr]

# In[92]:


# get activations
act_val_all = get_activations_batch(model_unet, -2, data_val_overlap.im_patches, 20, verbose=True)

# retain only activation weights for which there is a ground truth
act_val_all = remove_overlap(data_val.imgs, act_val_all, patch_size=64, stride=32)
act_val = act_val_all[pred_t_val]

# In[93]:


# get activations for testing Density Forest
act_test = get_activations_batch(model_unet, -2, data_test_overlap.im_patches, 20, verbose=True)

# remove test activations overlap
act_test = remove_overlap(data_test.imgs, act_test, patch_size=64, stride=32)
act_test = np.concatenate(np.concatenate(act_test))

# In[94]:


# get balanced data subset to show in figure
tsne_pts_per_class = 200
dataset_subset_indices = get_balanced_subset_indices(data_test.gt_patches.flatten(), 
                                                     np.arange(1, 9), pts_per_class=tsne_pts_per_class)
dataset_subset_indices = np.concatenate(dataset_subset_indices)

# In[ ]:


# t-SNE visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_all = tsne.fit_transform(act_test[dataset_subset_indices])
tsne_y = data_test.gt_patches.flatten()[dataset_subset_indices]

# In[ ]:


# plot
_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_axis_off()
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors, class_to_remove=class_to_remove)
plt.savefig("../Figures/Zurich/tSNE/t-SNE_wo_cl" + str(class_to_remove) + "_before_PCA.pdf",
            bbox_inches='tight', pad_inches=0)

# In[ ]:


# create density tree for activation weights of training data
# PCA
pca = decomposition.PCA(n_components=.95)
pca.fit(act_test)  # fit on training set without background pixels
n_components = np.alen(pca.explained_variance_ratio_)
print("Variance explained by first %i components: %.2f" % (
    n_components, sum(pca.explained_variance_ratio_)))

# transform training activations
act_train_all = pca.transform(np.concatenate(np.concatenate(act_train_all)))
act_train = pca.transform(act_train)

act_val_all = pca.transform(np.concatenate(np.concatenate(act_val_all)))
act_val = pca.transform(act_val)


# transform test set activations
act_test = pca.transform(act_test)

# In[ ]:


# Plot cumulative explained variance
fig = plt.figure()
plt.scatter(np.arange(n_components), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative sum of explained variance")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.savefig("../Figures/Zurich/PCA/ZH_pca_components_wo_cl_" + str(class_to_remove) + ".pdf",
            bbox_inches='tight', pad_inches=0)

# In[ ]:


# t-SNE visualization after PCA
tsne_all = tsne.fit_transform(act_test[dataset_subset_indices])
# tsne without unseen class
tsne_train = tsne_all[tsne_y != class_to_remove]

# In[ ]:


# plot
_, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors, class_to_remove=class_to_remove)
ax.set_axis_off()
plt.savefig("../Figures/Zurich/tSNE/t-SNE_wo_cl" + str(class_to_remove) + "_after_PCA.pdf",
            bbox_inches='tight', pad_inches=0)

# In[39]:


# plot first 3 PCA components
plot_pts_3d(act_test[:, :3], data_test.gt_patches.flatten(), classes_to_keep, colors,
            class_to_remove=class_to_remove, subsample_pct=.0003,
            s_name='../Figures/Zurich/PCA/pca_components_3d_' + str(names[class_to_remove]) + '.pdf')

print("Variance explained by first 3 components: %.2f" % np.sum(pca.explained_variance_ratio_[:3]))

# In[40]:


# plot first 2 PCA components
_, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_axis_off()
plot_pts_2d(act_test[:, :2], data_test.gt_patches.flatten(), ax, classes_to_keep, colors,
            class_to_remove=class_to_remove, subsample_pct=.0005,
            s_name='../Figures/Zurich/PCA/pca_components_2d_' + str(names[class_to_remove]) + '.pdf')
print("Variance explained by first 2 components: %.2f" % np.sum(pca.explained_variance_ratio_[:2]))

# ## GMM

# In[41]:


best_degrees = [3, 5, 4, 3, 9, 5, 9, 3]

if paramsearch:
    tuned_parameters = [{'n_components': np.arange(3, 10), 'max_iter': [10000]}]
    # do parameter search
    ps_gmm = ParameterSearch(GaussianMixture, tuned_parameters, act_train, act_train_all,
                             pred_f_tr.flatten(), scorer_roc_probas_gmm, 
                             n_iter=3, verbosity=10, n_jobs=-1, subsample_train=.01, subsample_test=.01)
    ps_gmm.fit()
    best_params_gmm = ps_gmm.best_params
else:
    best_params_gmm = {'n_components': df_ps.loc[str(names[class_to_remove])]['gmm_n_components'], 'max_iter': 10000}
print(best_params_gmm)

# In[42]:


# Fit GMM
gmm = GaussianMixture(**best_params_gmm)
gmm.fit(draw_subsamples(act_train, .1))

# Predict
probas_gmm = gmm.predict_proba(act_test)
probas_gmm = get_acc_net_entropy(probas_gmm)

# In[43]:


# precision-recall curve
y_scores = -probas_gmm
precision_gmm, recall_gmm, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_gmm = metrics.auc(recall_gmm, precision_gmm)

# ROC
fpr_gmm, tpr_gmm, _ = metrics.roc_curve(y_true, y_scores)
auroc_gmm = metrics.roc_auc_score(y_true, y_scores)
plt.step(recall_gmm, precision_gmm)
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_gmm, pr_auc_gmm))

# In[44]:


def equalize_exp(im, n_iter):
    """ad-hoc contrast enhancement for GMM"""
    for _ in range(n_iter):
        im = np.exp(im)
        im = exposure.equalize_hist(im)
    return im

# In[45]:


# visualization
probas_patches_gmm = np.reshape(probas_gmm, np.shape(data_test.gt_patches))
probas_patches_gmm -= np.min(probas_patches_gmm)
probas_patches_gmm /= np.max(probas_patches_gmm)

# show image of DF uncertainty vs. max margin uncertainty
acc_im_gmm = convert_patches_to_image(data_test.imgs, probas_patches_gmm[..., np.newaxis], 64, 64)
for img_idx in range(len(data_test.imgs)):
    acc_im_gmm_ = acc_im_gmm[img_idx]
    acc_im_gmm_ = equalize_exp(acc_im_gmm_, 10)
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_gmm_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_gmm_, f_name, dpi=my_dpi)

# 
# # SVM

# In[46]:


act_train_svm = preprocessing.scale(act_train)
act_train_all_svm = preprocessing.scale(act_train_all)
act_val_all_svm = preprocessing.scale(act_val_all)
act_test_svm = preprocessing.scale(act_test)

# In[47]:


# parameter search

if paramsearch:
    tuned_parameters = [{'kernel': ['rbf'],
                         'nu': [1e-4, 1e-3, 1e-2, 1e-1, 5e-1],
                         'degree':[1]
                         },
                        {'kernel': ['poly'],
                         'degree': np.arange(1, 4),
                         'nu': [1e-4, 1e-3, 1e-2, 1e-1, 5e-1],
                         'max_iter': [10000]}]

    # do parameter search
    ps_svm = ParameterSearch(svm.OneClassSVM, tuned_parameters, act_train_svm, act_train_all,
                             pred_f_tr.flatten(), scorer_roc_probas_svm, n_iter=5,
                             verbosity=11, n_jobs=-1, subsample_train=.0001, subsample_test=.001)
    ps_svm.fit()
    best_params_svm = ps_svm.best_params
else:
    best_params_svm = {'kernel': df_ps.loc[str(names[class_to_remove])]['oc-svm_k'], 
                       'degree': df_ps.loc[str(names[class_to_remove])]['oc-svm_deg'], 
                       'nu': df_ps.loc[str(names[class_to_remove])]['oc_svm_nu'],
                       'max_iter':10000}
    
print(best_params_svm)

# In[48]:


# Fit SVM
clf_svm = svm.OneClassSVM(**best_params_svm)
clf_svm.fit(draw_subsamples(act_train_svm, .001))

# In[49]:


# predict
probas_svm = clf_svm.decision_function(act_test_svm[dataset_subset_indices])
probas_svm -= np.min(probas_svm)
probas_svm /= np.max(probas_svm)

# In[50]:


probas_svm = clf_svm.decision_function(act_test_svm)
probas_svm -= np.min(probas_svm)
probas_svm /= np.max(probas_svm)

# In[51]:


# metrics

y_scores = -probas_svm[:]
# PR
precision_svm, recall_svm, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_svm = metrics.auc(recall_svm, precision_svm)

# ROC
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_true, y_scores)
auroc_svm = metrics.roc_auc_score(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_svm, pr_auc_svm))

# In[52]:


# visualization
probas_patches_svm = np.reshape(probas_svm, np.shape(data_test.gt_patches))

# show image of DF uncertainty vs. max margin uncertainty
acc_im_svm = convert_patches_to_image(data_test.imgs, probas_patches_svm[..., np.newaxis], 64, 64)
for img_idx in range(len(data_test.imgs)):
    acc_im_svm_ = exposure.equalize_hist(acc_im_svm[img_idx])
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_svm_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_svm_, f_name, dpi=my_dpi)

# #### Visualize Kernels

# In[53]:


subset_ind = get_balanced_subset_indices(data_train.gt_patches.flatten(), classes_to_keep, pts_per_class=150)

subsample = act_train_all_svm[np.concatenate(subset_ind)]

# In[54]:


# RBF
K = RBF()
K_X = K.__call__(subsample)
K_X = exposure.equalize_hist(K_X)
f_name = "../Figures/Zurich/Kernels/Kernel_RBF_wo_cl_" + str(class_to_remove) + ".jpg" 
export_figure_matplotlib(K_X, f_name, dpi=my_dpi)

# polynomial
for deg in [1, 2, 3]:
    K_X = metrics.pairwise.polynomial_kernel(subsample, degree=deg)
    # contrast stretching
    p2, p98 = np.percentile(K_X, (2, 98))
    K_X = exposure.rescale_intensity(K_X, in_range=(p2, p98))

    f_name = "../Figures/Zurich/Kernels/Kernel_poly_wo_cl_" + str(class_to_remove) + "_deg_" + str(deg) + ".jpg"
    export_figure_matplotlib(K_X, f_name, dpi=my_dpi)

# ## Density Forest

# In[55]:


# Create DensityForest instance
clf_df = DensityForest(max_depth=2, min_subset=.1, n_trees=100,
                       subsample_pct=.1, n_jobs=-1, verbose=10,
                       ig_improvement=.4)

# In[56]:


# fit to training data
clf_df.fit(tsne_train)

# In[57]:


# Show ellipses on plot
_, axes = plt.subplots(2, 2, figsize=(15, 15)) 
for i in range(4):
    plot_pts_2d(tsne_all, tsne_y, axes[int(i/2)][np.mod(i, 2)], classes_to_keep, 
                colors, class_to_remove=class_to_remove)
    axes[int(i/2)][np.mod(i, 2)].set_axis_off()
    covs, means = get_clusters(clf_df.root_nodes[i], [], [])
    plot_ellipses(axes[int(i / 2)][np.mod(i, 2)], means, covs)

# In[58]:


# export some ellipses for GIF

for i in range(10):
    _, ax = plt.subplots(1, 1, figsize=(8, 8)) 
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors, 
                class_to_remove=class_to_remove, names=names)
    covs, means = get_clusters(clf_df.root_nodes[i], [], [])
    plot_ellipses(ax, means, covs)
    plt.axis('off')
    plt.savefig("../Figures/Zurich/GIF/TSNE_act_wo_cl" + str(class_to_remove) + "_"+str(i)+".pdf", 
                bbox_inches='tight', pad_inches=0)
    plt.close()

# In[59]:


# plot

# get probabilities for all images
probas_df = np.log(clf_df.predict(tsne_all))

# indices of correctly / wrongly predicted points
pred_f = pred_f_te.flatten()[dataset_subset_indices]
pred_t = pred_t_te.flatten()[dataset_subset_indices]

# plot colors
probas_df_c = imgs_stretch_eq(probas_df[np.newaxis, ..., np.newaxis])[0, ..., 0]
colors_plt = plt.cm.YlOrRd(1 - probas_df_c)[..., :3]

# threshold for second plot
c_thresh_t = plt.cm.GnBu((probas_df < np.sort(probas_df)[tsne_pts_per_class])*255)[:, :3]

c_thresh_f = plt.cm.GnBu((probas_df > np.sort(probas_df)[tsne_pts_per_class])*255)[:, :3]

# plot correctly predicted points (o marker)
fig, axes = plt.subplots(1, 2, figsize=(20, 10)) 
axes[0].scatter(tsne_all[:, 0][pred_t], tsne_all[:, 1][pred_t], c=colors_plt[pred_t])
axes[1].scatter(tsne_all[:, 0][pred_t], tsne_all[:, 1][pred_t], c=c_thresh_t[pred_t])

# plot incorrectly predicted points (x marker)
axes[0].scatter(tsne_all[:, 0][pred_f], tsne_all[:, 1][pred_f], c=colors_plt[pred_f], marker='x')
axes[1].scatter(tsne_all[:, 0][pred_f], tsne_all[:, 1][pred_f], c=c_thresh_f[pred_f], marker='x')
[axes[i].legend(['Seen points', 'Novel points']) for i in range(2)]
[axes[i].set_axis_off() for i in range(2)]
extent = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("../Figures/Zurich/GIF/probas.pdf", bbox_inches=extent, pad_inches=0)

# ### Fit on real data

# In[60]:


# parameter search
default_params = {'n_trees': 5, 'n_max_dim': 0, 'n_jobs': -1, 
                  'verbose': 0, 'subsample_pct': .0002, 'min_subset': 1e-3}

if paramsearch:
    """search for best hyperparameters"""
    tuned_params = [{'max_depth': [1, 2, 3],
                     'ig_improvement': [-np.infty, 0, .4, .7]
                    }]

    # do parameter search
    ps_df = ParameterSearch(DensityForest, tuned_params, act_train, act_train_all,
                            pred_f_tr.flatten(), scorer_roc_probas_df,
                            n_iter=3, verbosity=11, n_jobs=1, subsample_train=1, 
                            subsample_test=.001, default_params=default_params)

    print("Testing %i combinations %i times" % (len(ps_df.combinations), ps_df.n_iter))
    print(ps_df.combinations)
    ps_df.fit()
    print(ps_df.best_params)
    
    # Create DensityForest instance
    best_params_df = ps_df.best_params
    
else:
    """use previously found hyperparameters"""
    best_params_df = {'max_depth': df_ps.loc[str(names[class_to_remove])]['df_depth'],
                      'ig_improvement': df_ps.loc[str(names[class_to_remove])]['df_min_ig']}
    
    
print(best_params_df)
default_params['verbose'] = 1
default_params['batch_size'] = 10000

# In[61]:


# fit DF with best found parameters
clf_df = DensityForest(**best_params_df, **default_params)
clf_df.fit(act_train)

# get probabilities for all images
probas_df = clf_df.predict(act_test)

# ### Post-Treatment

# In[62]:


# reshape probas to (n_patches, patch_size, patch_size)
probas_patches_df = np.reshape(probas_df, np.shape(data_test.gt_patches))

# transformations
probas_patches_df -= np.nanmin(probas_patches_df)
probas_patches_df /= np.nanmax(probas_patches_df)

# ### Metrics

# In[63]:


# metrics
y_scores = -probas_df

# PR
precision_df, recall_df, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_df = metrics.auc(recall_df, precision_df)

# ROC
fpr_df, tpr_df, _ = metrics.roc_curve(y_true, y_scores)
auroc_df = metrics.roc_auc_score(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_df, pr_auc_df))

# In[66]:


# visualization
probas_patches_df = np.reshape(probas_df, np.shape(data_test.gt_patches))
probas_patches_df -= np.min(probas_patches_df)
probas_patches_df /= np.max(probas_patches_df)

# show image of DF uncertainty vs. max margin uncertainty
acc_im_df = convert_patches_to_image(data_test.imgs, probas_patches_df[..., np.newaxis], 64, 64)
for img_idx in range(len(data_test.imgs)):
    #acc_im_df_ = exposure.equalize_hist(acc_im_df[img_idx])
    # TODO try log for contrast enhancement
    acc_im_df_= exposure.equalize_hist(np.log(acc_im_df[img_idx]+1e-10))
    f_name = base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_df_im_" + str(img_idx) + ".jpg"
    export_figure_matplotlib(acc_im_df_, f_name, dpi=my_dpi)

# ## Plot Results

# In[67]:


# Precision-Recall Curve
# order according to increasing score
scores_pr = [pr_auc_msr, pr_auc_margin, pr_auc_entropy, pr_auc_dropout, pr_auc_gmm, pr_auc_svm, pr_auc_df]

recalls = [recall_msr, recall_margin, recall_entropy, recall_dropout, recall_gmm, recall_svm, recall_df]
precisions = [precision_msr, precision_margin, precision_entropy, precision_dropout, 
              precision_gmm, precision_svm, precision_df]

names_methods = np.array(['MSR', 'Margin', 'Entropy', 'Dropout', 'GMM', 'OC SVM', 'DF'])
scores_order = np.argsort(scores_pr)
colors_lines = plt.cm.rainbow(np.linspace(0, 1, len(scores_pr)))[:, :3]

# plot
fig = plt.figure(figsize=(6, 6))
for i in scores_order:
    plt.step(recalls[i], precisions[i], where='post', c=colors_lines[i])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.legend([str.format('%s: %.2f') % (names_methods[i], scores_pr[i]) for i in scores_order], title="PR AUC")
plt.savefig("../Figures/Zurich/Metrics/PR_pred_wo_cl_" + str(class_to_remove) + ".pdf", 
            bbox_inches='tight', pad_inches=0)

# In[68]:


# ROC
# order according to increasing score
scores_auc = [auroc_msr, auroc_margin, auroc_entropy, auroc_dropout, auroc_gmm, auroc_svm, auroc_df]
fprs = [fpr_msr, fpr_margin, fpr_entropy, fpr_dropout, fpr_gmm, fpr_svm, fpr_df]
tprs = [tpr_msr, tpr_margin, tpr_entropy, tpr_dropout, tpr_gmm, tpr_svm, tpr_df]
scores_order = np.argsort(scores_auc)
colors_lines = plt.cm.rainbow(np.linspace(0, 1, len(scores_auc)))[:, :3]

# plot
fig = plt.figure(figsize=(6, 6))
for i in scores_order:
    plt.plot(fprs[i], tprs[i], c=colors_lines[i])

plt.plot([0, 1], [0, 1], '--', c='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.legend([str.format('%s: %.2f') % (names_methods[i], scores_auc[i]) for i in scores_order], title="AUROC")
plt.savefig("../Figures/Zurich/Metrics/ROC_pred_wo_cl_" + str(class_to_remove) + ".pdf", 
            bbox_inches='tight', pad_inches=0)

# In[69]:


# write best hyperparameters to CSV file
best_params = {'gmm_n_components': best_params_gmm['n_components'], 
               'oc-svm_k': best_params_svm['kernel'], 
               'oc-svm_deg': best_params_svm['degree'], 
               'oc_svm_nu': best_params_svm['nu'], 
               'df_depth': best_params_df['max_depth'], 
               'df_min_ig': best_params_df['ig_improvement']}

# In[70]:


# write results to CSV files
# hyperparameters
df_ps = pd.read_csv('models_out/hyperparams.csv', index_col=0)
df2 = pd.DataFrame(best_params, index=[str(names[class_to_remove])])
df_ps = df_ps.append(df2)
df_ps = df_ps[~df_ps.index.duplicated(keep='last')]  # avoid duplicates
df_ps.to_csv('models_out/hyperparams.csv')

# AUROC
df_auroc = pd.read_csv('models_out/auroc_all.csv', index_col=0)
df2 = pd.DataFrame({str(names[class_to_remove]): scores_auc}, index = names_methods).T
df_auroc = df_auroc.append(df2)
df_auroc = df_auroc[~df_auroc.index.duplicated(keep='last')]  # avoid duplicates
df_auroc.to_csv('models_out/auroc_all.csv')


# PR AUC
df_aucpr = pd.read_csv('models_out/aucpr_all.csv', index_col=0)
df2 = pd.DataFrame({str(names[class_to_remove]): scores_pr}, index = names_methods).T
df_aucpr = df_aucpr.append(df2)
df_aucpr = df_aucpr[~df_aucpr.index.duplicated(keep='last')]  # avoid duplicates
df_aucpr.to_csv('models_out/aucpr_all.csv')

# In[71]:


# load auroc df with previously saved results
df_auroc = pd.read_csv('models_out/auroc_all.csv', index_col=0)

# In[72]:


# reorder by class names
df_auroc = df_auroc.reindex(names[1:])
df_auroc.to_csv('models_out/auroc_all.csv')

# In[73]:


# show latex table of AUROC metrics for all methods
print(df_auroc.round(2).to_latex())

# In[74]:


# show best method for each left-out class
df_auroc.T.idxmax()

# In[75]:


# show mean aourc for each method
df_auroc.mean().round(2)

# In[76]:


df_ps

# In[77]:


# show mean aourc for each method
print(df_ps.to_latex())

# # Interesting Objects

# In[78]:


import matplotlib.patches as patches

# In[79]:


my_dpi = 255

# In[80]:


def export_figure_matplotlib(arr, f_name=None, dpi=255, resize_fact=1, plt_show=False, rect=None):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    :param rect: rectangle to overlay over image
    """
    # plot
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap='RdYlGn')

    # add rectangle overlay
    if rect is not None:
        ax.add_patch(rect)

    # save figure
    if f_name is not None:
        plt.savefig(f_name, dpi=(dpi * resize_fact))

    if plt_show:
        plt.show()
    else:
        plt.close()
    return fig

# In[81]:


def export_particularity(dataset, img_idx, x, y, width, height, confs_patches=None, names=None, dir_out=None, idx=0):
    """
    export detail view of interesting image: image with blue rectangle, cropped region, cropped GT, cropped conf images
    """
    # Image
    class_to_remove = dataset.class_to_remove
    im = dataset.imgs[img_idx][..., :3]
    max_x = np.mod(im.shape[0], 32)
    max_y = np.mod(im.shape[1], 32)
    im = im[:-max_x, :-max_y]
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='b', alpha=.5)
    f_name = dir_out+ 'im_' + str(img_idx) + '_obj_' + str(idx) + '_im.jpg'
    fig = export_figure_matplotlib(im, dpi=my_dpi, rect=rect, f_name=f_name)
    plt.close()
    
    # Cropped Image
    im_crop = im[y:(y + height), x:(x + width)]
    f_name = dir_out+ 'im_' + str(img_idx) + '_obj_' + str(idx) + '_im_crop.jpg'
    fig = export_figure_matplotlib(im_crop, dpi=my_dpi, f_name=f_name)
    plt.close()
    
    # GT
    rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor='b', facecolor='None')
    im_gt = gt_label_to_color(dataset.gt[img_idx], colors)*255
    max_x = np.mod(im_gt.shape[0], 32)
    max_y = np.mod(im_gt.shape[1], 32)
    im_gt = im_gt[:-max_x, :-max_y]
    f_name = dir_out+ 'im_' + str(img_idx) + '_obj_' + str(idx) + '_gt.jpg'
    fig = export_figure_matplotlib(im_gt, dpi=my_dpi, rect=rect, f_name=f_name)
    plt.close()

    if confs_patches is not None:
        for idx_conf,conf_patches in enumerate(confs_patches):
            conf_im = convert_patches_to_image(dataset.imgs, conf_patches[..., np.newaxis], 64, 64)
            conf_im = exposure.equalize_hist(conf_im[img_idx])
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='None')
            f_name = dir_out+ 'im_' + str(img_idx) + '_obj_' + str(idx) + '_' + names[idx_conf] + '_wo_cl_'+ str(class_to_remove) + '.jpg'
            export_figure_matplotlib(conf_im, dpi=my_dpi, rect=rect, f_name=f_name)
            plt.close()

# In[82]:


names_confs_patches = ['msr', 'df']
confs_patches = [probas_patches_msr, probas_patches_df]

# In[83]:


dir_out = '../Figures/Zurich/Detail/im1/'
export_particularity(data_test, 1, 270, 660, 200, 200, dir_out=dir_out, confs_patches=confs_patches, names=names_confs_patches, idx=1)

# In[84]:


dir_out = '../Figures/Zurich/Detail/im3/'
confs_patches = [probas_patches_msr, probas_patches_df] 

# In[85]:


dir_out = '../Figures/Zurich/Detail/im4/'
export_particularity(data_test, 4, 670, 550, 450, 320, dir_out=dir_out, confs_patches=confs_patches, names=names_confs_patches, idx=1)

# # Plot confidence in t-SNE

# In[86]:


names_methods

# In[87]:


pred_f_tsne = data_test.gt_patches == class_to_remove
pred_t_tsne = data_test.gt_patches != class_to_remove

# In[88]:


# indices of correctly / wrongly predicted points
pred_f = pred_f_tsne.flatten()[dataset_subset_indices]
pred_t = pred_t_tsne.flatten()[dataset_subset_indices]
probas_methods = [probas_msr, probas_margin, probas_entropy, probas_dropout.flatten(), probas_gmm, 
                  np.squeeze(probas_svm), np.squeeze(probas_df)]
base_dir = "../Figures/Zurich/Im_cert/cl_" + str(class_to_remove) + "/"

names_methods = ['net_msr', 'net_margin', 'net_entropy', 'dropout', 'gmm', 'svm', 'df']
for proba, name in zip(probas_methods, names_methods):
    probas_tsne = proba[dataset_subset_indices]

    # colors
    probas_df_c = probas_tsne
    probas_df_c = exposure.equalize_hist(probas_tsne)
    probas_df_c -= np.min(probas_df_c)
    probas_df_c /= np.max(probas_df_c)
    colors_plt = plt.cm.RdYlGn(probas_df_c)[..., :3]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
    # indicators of least confident points    
    # plot correctly predicted points (o marker)

    ax.scatter(tsne_all[:, 0][pred_t], tsne_all[:, 1][pred_t], c=colors_plt[pred_t], alpha=.3)

    # plot incorrectly predicted points (x marker)
    ax.scatter(tsne_all[:, 0][pred_f], tsne_all[:, 1][pred_f], c=colors_plt[pred_f], edgecolors='black',
               linewidths=1, s=90)
    ax.set_axis_off()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    f_name = base_dir + "t-SNE_wo_cl_" + str(class_to_remove) + "_" + str(name) + ".pdf"
    plt.savefig(f_name, bbox_inches=extent, pad_inches=0)
    ax.legend(['Seen points', 'Novel points'])
    plt.close()

# In[ ]:


# indices of correctly / wrongly predicted points
pred_f = pred_f_tsne.flatten()[dataset_subset_indices]
pred_t = pred_t_tsne.flatten()[dataset_subset_indices]
probas_methods = [probas_msr, probas_margin, probas_entropy, probas_dropout.flatten(), probas_gmm, 
                  np.squeeze(probas_svm), np.squeeze(probas_df)]
base_dir = "../Figures/Zurich/Im_cert/cl_" + str(class_to_remove) + "/"

names_methods = ['net_msr', 'net_margin', 'net_entropy', 'dropout', 'gmm', 'svm', 'df']
for proba, name in zip(probas_methods, names_methods):
    probas_tsne = proba[dataset_subset_indices]

    # colors
    colors_plt = plt.cm.RdYlGn((probas_tsne > np.sort(probas_tsne)[tsne_pts_per_class]) * 255)

    # plot correctly predicted points (o marker)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10)) 
    ax.scatter(tsne_all[:, 0][pred_t], tsne_all[:, 1][pred_t], c=colors_plt[pred_t], alpha=.2)

    # plot incorrectly predicted points (x marker)
    ax.scatter(tsne_all[:, 0][pred_f], tsne_all[:, 1][pred_f], c=colors_plt[pred_f], edgecolors='black',
               linewidths=1, s=90)
    
    ax.set_axis_off()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    f_name = base_dir + "t-SNE_wo_cl_" + str(class_to_remove) + "_" + str(name) + "_leastcert.pdf"
    plt.savefig(f_name, bbox_inches=extent, pad_inches=0)
    ax.legend(['Seen points', 'Novel points'])
    plt.close()
