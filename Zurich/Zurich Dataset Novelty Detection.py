
# coding: utf-8

# # Zurich Land Cover Classification
# 
# This script presents a visualization of training a U-Net classifier on 7 out of 8 available land cover classes of the Zurich dataset, and detecting the unseen class using a Density Forest.

# ## Import Libraries

# In[1]:


# python libraries
import sys
from sklearn.manifold import TSNE
from sklearn import decomposition
from keras.models import load_model
from tensorflow.python.client import device_lib
from helpers.plots import *
from helpers.data_loader import *
from helpers.parameter_search import *
from density_forest.helpers import *
from baselines.helpers import *
from keras_helpers.unet import *
from keras_helpers.callbacks import *

class_to_remove = int(sys.argv[1])
paramsearch = False  # search for best hyperparameters
my_dpi=255 # dpi of my screen, for image exporting

# data frame with previously found optimal hyperparameters
df_ps = pd.read_csv('models_out/hyperparams.csv', index_col=0) 
df_auroc = pd.read_csv('models_out/auroc_all.csv', index_col=0)
df_aucpr = pd.read_csv('models_out/aucpr_all.csv', index_col=0)


# # Load Data

# In[2]:


path = os.getcwd()

# data without overlap
print("loading data")
data_train = ZurichLoader(path, 'train', class_to_remove=class_to_remove)
data_test = ZurichLoader(path, 'test', class_to_remove=class_to_remove)

print("loading data with overlap")
# data with overlap, for prediction
data_train_overlap = ZurichLoader(path, 'train', stride=32, inherit_loader=data_train)
data_test_overlap = ZurichLoader(path, 'test', stride=32, inherit_loader=data_test)

# class names and colors
names = data_train.names
colors = data_train.colors
n_classes = 8
classes_to_keep = np.asarray([x for x in range(1, n_classes + 1) if x != class_to_remove])
names_keep = np.asarray(names)[classes_to_keep]
print("classes to keep: " + str(names_keep))


# In[7]:


colors_unseen = colors.copy()
colors_unseen[2] = [.98, 0, .02]
colors_unseen[class_to_remove] = [.5, .5, .5]


# In[3]:


# export patch illustration
export_pad(data_train.imgs[0], 128, 64, s_name='../Figures/Zurich/im_padding.pdf', cmap=cm.rainbow)


# In[4]:


for dataset, offset in zip([data_train, data_test], [0, 15]):
    for im_idx, im in enumerate(dataset.imgs):
        im = im[..., :3]
        f_name = "../Figures/Zurich/Im/Im_" + str(im_idx + offset) + ".jpg"
        export_figure_matplotlib(im, f_name, dpi=my_dpi)


# In[5]:


for dataset, offset in zip([data_train, data_test], [0, 15]):
    for gt_idx, gt in enumerate(dataset.gt):
        gt_col = gt_label_to_color(gt, colors)*255
        f_name = "../Figures/Zurich/Im/GT_" + str(gt_idx + offset) + ".jpg"
        export_figure_matplotlib(gt_col, f_name, dpi=my_dpi)


# In[6]:


pred_labels_tr, cnt_tr = np.unique(data_train.gt_patches.astype('int'), return_counts=True)
pred_labels_te, cnt_te = np.unique(data_test.gt_patches.astype('int'), return_counts=True)

cnt_tr = cnt_tr / np.sum(cnt_tr) * 100
cnt_te = cnt_te / np.sum(cnt_te) * 100




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


# In[13]:

# ## Retrieve Activations, PCA, t-SNE

pred_t_tr = y_pred_label_tr == data_train.gt_patches


# get activations for training Density Forest
act_train_all = get_activations_batch(model_unet, -2, data_train_overlap.im_patches, 20, verbose=True)

# retain only activation weights for which there is a ground truth
act_train_all = remove_overlap(data_train.imgs, act_train_all, patch_size=64, stride=32)
act_train = act_train_all[pred_t_tr]


# In[30]:


# get activations for testing Density Forest
act_test = get_activations_batch(model_unet, -2, data_test_overlap.im_patches, 20, verbose=True)

# remove test activations overlap
act_test = remove_overlap(data_test.imgs, act_test, patch_size=64, stride=32)
act_test = np.concatenate(np.concatenate(act_test))


# In[31]:


# get balanced data subset to show in figure
tsne_pts_per_class = 200
dataset_subset_indices = get_balanced_subset_indices(data_test.gt_patches.flatten(), 
                                                     np.arange(1, 9), pts_per_class=tsne_pts_per_class)
dataset_subset_indices = np.concatenate(dataset_subset_indices)


# In[32]:


# t-SNE visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_all = tsne.fit_transform(act_test[dataset_subset_indices])
tsne_y = data_test.gt_patches.flatten()[dataset_subset_indices]


# In[33]:


colors[-2] = [1., 1, 0] 


# In[34]:


# plot
_, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_axis_off()
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors_unseen, class_to_remove=class_to_remove)
plt.savefig("../Figures/Zurich/tSNE/t-SNE_wo_cl" + str(class_to_remove) + "_before_PCA.pdf",
            bbox_inches='tight', pad_inches=0)


# In[35]:


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

# transform test set activations
act_test = pca.transform(act_test)


# In[36]:


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


# In[37]:


# t-SNE visualization after PCA
tsne_all = tsne.fit_transform(act_test[dataset_subset_indices])
# tsne without unseen class
tsne_train = tsne_all[tsne_y != class_to_remove]


# In[155]:


# plot
_, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_axis_off()
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors_unseen, class_to_remove=class_to_remove)

plt.savefig("../Figures/Zurich/tSNE/t-SNE_wo_cl" + str(class_to_remove) + "_after_PCA.pdf",
            bbox_inches='tight', pad_inches=0)


# In[39]:


# plot first 3 PCA components
plot_pts_3d(act_test[:, :3], data_test.gt_patches.flatten(), classes_to_keep, colors_unseen,
            class_to_remove=class_to_remove, subsample_pct=.0003,
            s_name='../Figures/Zurich/PCA/pca_components_3d_' + str(names[class_to_remove]) + '.pdf')

print("Variance explained by first 3 components: %.2f" % np.sum(pca.explained_variance_ratio_[:3]))


# In[40]:


# plot first 2 PCA components
_, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_axis_off()
plot_pts_2d(act_test[:, :2], data_test.gt_patches.flatten(), ax, classes_to_keep, colors_unseen,
            class_to_remove=class_to_remove, subsample_pct=.0005,
            s_name='../Figures/Zurich/PCA/pca_components_2d_' + str(names[class_to_remove]) + '.pdf')
print("Variance explained by first 2 components: %.2f" % np.sum(pca.explained_variance_ratio_[:2]))