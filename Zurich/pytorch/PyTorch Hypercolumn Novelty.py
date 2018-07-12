
# coding: utf-8

# # PyTorch Hypercolumn CNN  for Zurich Dataset

# In[1]:


# import libraries
import pandas as pd
import os, sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook as tqdm
import numpy as np
import gc
from helpers_pytorch import *

# custom libraries
base_dir = '/raid/home/cwendl'  # for guanabana
sys.path.append(base_dir + '/SIE-Master/Code')  # Path to density Tree package
from helpers.data_loader import ZurichLoader
from helpers.helpers import *
from hypercolumn import HyperColumn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


# load datasets
base_dir = '/raid/home/cwendl'  # for guanabana
root_dir = base_dir + '/SIE-Master/Zurich'
patch_size = 64
class_to_remove = 3

# load data
dataset_train = ZurichLoader(root_dir, 'train', patch_size=patch_size, stride=patch_size, transform='augment', 
                             random_crop=True, class_to_remove=class_to_remove)

dataset_val = ZurichLoader(root_dir, 'val', patch_size=patch_size, stride=patch_size, 
                           class_to_remove=class_to_remove)

dataset_test = ZurichLoader(root_dir, 'test', patch_size=patch_size, stride=patch_size, 
                            class_to_remove=class_to_remove)

dataloader_train = DataLoader(dataset_train, batch_size=100, shuffle=True, num_workers=20)
dataloader_val = DataLoader(dataset_val, batch_size=100, shuffle=False, num_workers=20)
dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=False, num_workers=20)


# In[3]:


train_bool = False

model = HyperColumn(in_dim=4, out_dim=9, n_filters=32, patch_size=patch_size).cuda()
if train_bool:
    # Train network from scratch
    train(model, dataloader_train, dataloader_val, epochs=50, verbosity=1, plot=True)
    
    # save model
    state = {'model': model.state_dict(), 
             'n_epochs': 50,
             'loss_tr':0.
            }
    
    torch.save(state, 'models/model_wo_cl_' + str(class_to_remove) + '.pytorch')

else:  # load saved model
    state = torch.load('models/model_wo_cl_' + str(class_to_remove) + '.pytorch')
    model.load_state_dict(state['model'])


# In[4]:


# load data with overlap
dataset_train_overlap = ZurichLoader(root_dir, 'train', patch_size=patch_size, stride=int(patch_size/2), 
                                     inherit_loader=dataset_train)
dataset_val_overlap = ZurichLoader(root_dir, 'val', patch_size=patch_size, stride=int(patch_size/2), 
                                   inherit_loader=dataset_val)
dataset_test_overlap = ZurichLoader(root_dir, 'test', patch_size=patch_size, stride=int(patch_size/2), 
                                    inherit_loader=dataset_test)

dataloader_train_overlap = DataLoader(dataset_train_overlap, batch_size=100, shuffle=False, num_workers=40)
dataloader_val_overlap = DataLoader(dataset_val_overlap, batch_size=100, shuffle=False, num_workers=40)
dataloader_test_overlap = DataLoader(dataset_test_overlap, batch_size=100, shuffle=False, num_workers=40)


# In[5]:


# free GPU memory
while gc.collect():
    torch.cuda.empty_cache()


# In[6]:


# Predict

# train
preds_tr = predict_softmax(model, dataloader_train_overlap)
preds_tr = remove_overlap(dataset_train.imgs, preds_tr, np.arange(10), patch_size=patch_size, stride=int(patch_size/2))
preds_tr = np.concatenate(preds_tr)

# val
preds_val = predict_softmax(model, dataloader_val_overlap)
preds_val = remove_overlap(dataset_val.imgs, preds_val, np.arange(5), patch_size=patch_size, stride=int(patch_size/2))
preds_val = np.concatenate(preds_val)

# test
preds_te = predict_softmax(model, dataloader_test_overlap)
preds_te = remove_overlap(dataset_test.imgs, preds_te, np.arange(5), patch_size=patch_size, stride=int(patch_size/2))
preds_te = np.concatenate(preds_te)


# In[7]:


# free GPU memory
while gc.collect():
    torch.cuda.empty_cache()


# In[8]:


# get labels
pred_labels_tr = get_y_pred_labels(preds_tr, class_to_remove=class_to_remove, background=False)
pred_labels_val = get_y_pred_labels(preds_val, class_to_remove=class_to_remove, background=False)
pred_labels_te = get_y_pred_labels(preds_te, class_to_remove=class_to_remove, background=False)


# In[9]:


# get indices of correctly / incorrectly predicted pixels
# train
pred_t_tr = (dataset_train.gt_patches != class_to_remove) & (dataset_train.gt_patches != 0)
pred_f_tr = dataset_train.gt_patches == class_to_remove

# val
pred_t_val = (dataset_val.gt_patches != class_to_remove) & (dataset_val.gt_patches != 0)
pred_f_val = dataset_val.gt_patches == class_to_remove

# test
pred_t_te = (dataset_test.gt_patches != class_to_remove) & (dataset_test.gt_patches != 0)
pred_f_te = dataset_test.gt_patches == class_to_remove


# In[10]:


img_idx = 2

img = convert_patches_to_image(dataset_val.imgs, pred_labels_val[..., np.newaxis], img_idx, patch_size, patch_size, 0)

# pred
plt.figure(figsize=(8, 8))
plt.imshow(gt_label_to_color(img[..., 0], dataset_val.colors)*255)
plt.show()

# im
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
axes[0].imshow(dataset_val.imgs[img_idx][..., :3])
axes[1].imshow(gt_label_to_color(dataset_val.gt[img_idx], dataset_val.colors)*255)
plt.show()


# In[11]:


from baselines.helpers import *


# In[12]:


# class names and colors
names = dataset_train.names
colors = dataset_train.colors
n_classes = 9
classes_to_keep = np.asarray([x for x in range(1, n_classes) if x != class_to_remove])
names_keep = np.asarray(names)[classes_to_keep]
print("classes to keep: " + str(names_keep))


# In[13]:


# distribution of predicted label
pred_labels, pred_counts = np.unique(pred_labels_te[pred_f_te], return_counts=True)
pred_counts = pred_counts / sum(pred_counts) * 100

# visualization
fig = plt.figure(figsize=(7, 5))
plt.bar(pred_labels, pred_counts)
plt.xticks(np.arange(0, 10))
plt.ylim([0, 100])
plt.xlabel("Predicted Label")
plt.ylabel("Count [%]")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.title("Misclassified labels (mean MSR=%.2f)" % np.mean(get_acc_net_msr(preds_te[pred_f_te])))
plt.xticks(np.arange(len(names)), names, rotation=20)
plt.savefig("Figures/ZH_pred-count_wo_cl" + str(class_to_remove) + ".pdf",
            bbox_inches='tight', pad_inches=0)


# In[14]:


# TODO get weights
# p = [p for p in model.parameters()]
# for p_ in p:
    # print(p_.cpu().detach().numpy().shape)


# In[15]:


ones = np.ones(len(np.unique(dataset_train.gt_patches)))
ones[0] = 0
weight = torch.from_numpy(ones).float().cuda()
f_loss = nn.CrossEntropyLoss(weight=weight)
p = test(model, f_loss, dataloader_train, "Train", verbosity=1)
p = test(model, f_loss, dataloader_val, "Val", verbosity=1)
p = test(model, f_loss, dataloader_test, "Test", verbosity=1)


# In[16]:


# Accuracy measures for each class
y_preds = [pred_labels_tr, pred_labels_val, pred_labels_te]
datasets = [dataset_train, dataset_val, dataset_test]

aa_sets, oa_sets = [], []
for y_pred, y_true in zip(y_preds, datasets):
    y_pred_flattened = np.asarray(y_pred.flatten()).astype('int') 
    y_true_flattened = np.asarray(y_true.gt_patches.flatten()).astype('int') 

    # mask background and removed classes for evaluation metrics
    filter_items = (y_true_flattened != 0) & (y_true_flattened != class_to_remove)
    
    aa_set, _ = aa(y_true_flattened[filter_items], y_pred_flattened[filter_items])
    oa_set = oa(y_true_flattened[filter_items], y_pred_flattened[filter_items])
    print("OA: %.3f, AA: %.3f" % (oa_set, aa_set))  # slightly higher accuracy because of overlapping patches
    oa_sets.append(oa_set)
    aa_sets.append(aa_set)


# In[17]:


# TODO save CSV
# write metrics to CSV files
df_metrics = pd.read_csv('models/metrics_ND.csv', index_col=0)
accs = np.concatenate([[oa_sets[i], aa_sets[i]] for i in range(3)])  # [oa, aa] for tr, val, te
df2 = pd.DataFrame({str(names[class_to_remove]):accs},
                    index = ['OA Train', 'AA Train', 'OA Val', 'AA Val', 'OA Test', 'AA Test']).T
df_metrics = df_metrics.append(df2)
df_metrics = df_metrics[~df_metrics.index.duplicated(keep='last')]  # avoid duplicates
df_metrics.to_csv('models/metrics_ND.csv')
print((df_metrics*100).round(2).to_latex())


# # Network

# In[65]:


# precision-recall curves

# msr
y_scores = (-get_acc_net_msr(preds_te)).flatten()
y_true = pred_f_te.flatten()
precision_msr, recall_msr, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_msr = metrics.average_precision_score(y_true, y_scores)
auroc_msr = metrics.roc_auc_score(y_true, y_scores)
fpr_msr, tpr_msr, _ = metrics.roc_curve(y_true, y_scores)

# margin
y_scores = (-get_acc_net_max_margin(preds_te)).flatten()
precision_margin, recall_margin, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_margin = metrics.average_precision_score(y_true, y_scores)
auroc_margin = metrics.roc_auc_score(y_true, y_scores)
fpr_margin, tpr_margin, _ = metrics.roc_curve(y_true, y_scores)

# entropy
y_scores = (-get_acc_net_entropy(preds_te)).flatten()
precision_entropy, recall_entropy, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_entropy = metrics.average_precision_score(y_true, y_scores)
auroc_entropy = metrics.roc_auc_score(y_true, y_scores)
fpr_entropy, tpr_entropy, _ = metrics.roc_curve(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_msr, pr_auc_msr))
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_margin, pr_auc_margin))
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_entropy, pr_auc_entropy))


# In[61]:


# visualization
# MSR
probas_patches_msr = np.reshape((get_acc_net_msr(preds_te)).flatten(), np.shape(dataset_test.gt_patches))
probas_patches_msr -= np.min(probas_patches_msr)
probas_patches_msr /= np.max(probas_patches_msr)

# margin
probas_patches_margin = np.reshape((get_acc_net_max_margin(preds_te)).flatten(), np.shape(dataset_test.gt_patches))
probas_patches_margin -= np.min(probas_patches_margin)
probas_patches_margin /= np.max(probas_patches_margin)

# entropy
probas_patches_entropy = np.reshape((get_acc_net_entropy(preds_te)).flatten(), np.shape(dataset_test.gt_patches))
probas_patches_entropy -= np.min(probas_patches_entropy)
probas_patches_entropy /= np.max(probas_patches_entropy)

base_folder = "Figures"

# show images
for img_idx in range(len(dataset_test.imgs)):
    acc_im_msr = convert_patches_to_image(dataset_test.imgs, probas_patches_msr[..., np.newaxis], img_idx, 64, 64, 0)
    acc_im_msr = imgs_stretch_eq([acc_im_msr])[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(acc_im_msr[..., 0], cmap='RdYlGn')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_msr_im_" + str(img_idx) + ".pdf", 
                bbox_inches='tight', pad_inches=0)
    plt.close()
    
    acc_im_margin = convert_patches_to_image(dataset_test.imgs, probas_patches_margin[..., np.newaxis],
                                             img_idx, 64, 64, 0)
    acc_im_margin = imgs_stretch_eq([acc_im_margin])[0]
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.imshow(acc_im_margin[..., 0], cmap='RdYlGn')
    plt.savefig(base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_margin_im_" + str(img_idx) + ".pdf", 
                bbox_inches='tight', pad_inches=0)
    plt.close()
    
    acc_im_entropy = convert_patches_to_image(dataset_test.imgs, probas_patches_entropy[..., np.newaxis],
                                              img_idx, 64, 64, 0)
    acc_im_entropy = imgs_stretch_eq([acc_im_entropy])[0]
    plt.figure(figsize=(8, 8))
    plt.imshow(acc_im_entropy[..., 0], cmap='RdYlGn')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(base_folder + "/ZH_wo_cl_" + str(class_to_remove) + "_net_entropy_im_" + str(img_idx) + ".pdf", 
                bbox_inches='tight', pad_inches=0)
    plt.close()


# # Dropout

# In[33]:


def predict_softmax_w_dropout(model, dataloader_pred, n_iter):
    """
    Predict n_iter times using dropout using test time
    :param model:
    :param dataloader_pred:
    :param n_iter:
    :return:
    """
    # TODO test
    preds = []
    model.predict_dropout = True
    model.get_activations = False
    for _ in tqdm(range(n_iter)):
        with torch.no_grad():
            model.train()
            pred = []  # softmax activations
            for i_batch, (im, gt) in enumerate(dataloader_pred):
                im = im.cuda()
                output = model(im)
                pred.append(output.cpu())

        while gc.collect():
            torch.cuda.empty_cache()

        pred = np.concatenate([p.numpy() for p in pred])
        pred = np.transpose(pred, (0, 2, 3, 1))
        preds.append(pred)
    model.predict_dropout = False
    return np.asarray(preds)


# In[40]:


from tqdm import tqdm_notebook as tqdm


# In[52]:


imps = []
for iters in [10]:  # [2, 5, 20, 50]:
    preds = predict_softmax_w_dropout(model, dataloader_test, iters)
    # multiple models 
    ent1 = get_acc_net_entropy(np.mean(preds, 0))
    
    # one model
    ent2 = get_acc_net_entropy(preds[0])
    imp_fact1 = np.mean(ent1[pred_t_te]) / np.mean(ent2[pred_t_te])
    imp_fact2 = np.mean(ent1[pred_f_te]) / np.mean(ent2[pred_f_te])
    imp_imp_f = (imp_fact1 / imp_fact2) - 1  # how much the wrong regions get more uncertain wrt correct regions
    imps.append(imp_imp_f)
    
    print("%i iterations: f1=%.5f, f2=%.5f, f3=%.5f" % (iters, imp_fact1, imp_fact2, imp_imp_f))


# In[54]:


y_scores = -get_acc_net_entropy(np.mean(preds, 0)).flatten()


# In[57]:


# metrics
y_true = pred_f_te.flatten()

# PR
precision_gmm, recall_gmm, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_gmm = metrics.auc(recall_gmm, precision_gmm)

# ROC
fpr_gmm, tpr_gmm, _ = metrics.roc_curve(y_true, y_scores)
auroc_gmm = metrics.roc_auc_score(y_true, y_scores)
plt.step(recall_gmm, precision_gmm)
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_gmm, pr_auc_gmm))


# In[50]:


# clear CUDA storage
while gc.collect():
    torch.cuda.empty_cache()


# # Retrieve Activations

# In[67]:


act_val = get_activations(model, dataloader_val_overlap, 64)


# In[68]:


del dataset_val_overlap, dataloader_val_overlap


# ## PCA

# In[69]:


# python libraries
from sklearn.manifold import TSNE
from sklearn import decomposition, svm, preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics

# custom libraries
from helpers.parameter_search import *
from density_forest.density_forest import *
from density_forest.plots import *
from density_forest.helpers import *


# In[70]:


print(np.shape(draw_subsamples(act_val, .1)))
print(np.shape(np.concatenate(np.concatenate(draw_subsamples(act_val, .1)))))


# In[71]:


ind_bal = get_balanced_subset_indices(dataset_test.gt_patches.flatten(), np.arange(1, 9)).flatten()


# In[72]:


# create density tree for activation weights of training data
# PCA
pca = decomposition.PCA(n_components=.95)

# TODO fit on training set without background pixels
pca.fit(draw_subsamples(np.concatenate(np.concatenate(draw_subsamples(act_val, .1)))))  
n_components = np.alen(pca.explained_variance_ratio_)
print("Variance explained by first %i components: %.2f" % (
    n_components, sum(pca.explained_variance_ratio_)))


# In[73]:


# Plot cumulative explained variance
fig = plt.figure()
plt.scatter(np.arange(n_components), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative sum of explained variance")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.savefig("Figures/pca_components_wo_cl_" + str(class_to_remove) + ".pdf",
            bbox_inches='tight', pad_inches=0)


# In[74]:


act_val = pca.transform(np.concatenate(np.concatenate(act_val)))[..., :10]


# In[75]:


# get other activations

# test
act_test = get_activations(model, dataloader_test_overlap, 64)
print(np.shape(act_test))

# PCA 
act_test = pca.transform(np.concatenate(np.concatenate(act_test)))[..., :10]
print(np.shape(act_test))
del dataset_test_overlap, dataloader_test_overlap


# In[76]:


# t-SNE visualization
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_all = tsne.fit_transform(act_test[ind_bal])
tsne_y = dataset_test.gt_patches.flatten()[ind_bal]


# In[77]:


# plot
_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_axis_off()
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep, colors, class_to_remove=class_to_remove)
plt.savefig("Figures/t-SNE_" + str(names[class_to_remove]).lower().replace(" ", "") + "_after_PCA.pdf",
            bbox_inches='tight', pad_inches=0)


# In[78]:


# clear CUDA storage
while gc.collect():
    torch.cuda.empty_cache()


# In[79]:


# get other activations

# train
act_train = get_activations(model, dataloader_train_overlap, 64, pca)
print(act_train.shape)


# In[80]:


del dataset_train_overlap, dataloader_train_overlap


# In[81]:


# clear CUDA storage
while gc.collect():
    torch.cuda.empty_cache()


# # Confidence Estimation

# # GMM

# In[82]:


# Fit GMM
gmm = GaussianMixture(n_components=6, max_iter=10000)
gmm.fit(draw_subsamples(act_train[pred_t_tr.flatten()], .01))

# Predict
probas_gmm = gmm.predict_proba(act_test)
probas_gmm = get_acc_net_entropy(probas_gmm)


# In[83]:


# metrics
y_true = pred_f_te.flatten()
y_scores = -probas_gmm

# PR
precision_gmm, recall_gmm, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_gmm = metrics.auc(recall_gmm, precision_gmm)

# ROC
fpr_gmm, tpr_gmm, _ = metrics.roc_curve(y_true, y_scores)
auroc_gmm = metrics.roc_auc_score(y_true, y_scores)
plt.step(recall_gmm, precision_gmm)
print("AUROC: %.2f, PR AUC: %.2f" % (auroc_gmm, pr_auc_gmm))


# # SVM

# In[84]:


act_train_svm = preprocessing.scale(act_train)
#act_val_svm = preprocessing.scale(act_val)
act_test_svm = preprocessing.scale(act_test)


# In[85]:


# Fit SVM
clf_svm = svm.OneClassSVM(kernel='rbf', max_iter=10000)
clf_svm.fit(draw_subsamples(act_train_svm[pred_t_tr.flatten()], .001))


# In[ ]:


# predict
probas_svm = clf_svm.decision_function(act_test_svm)
probas_svm -= np.min(probas_svm)
probas_svm /= np.max(probas_svm)


# In[ ]:


# metrics

y_scores = -probas_svm[:]
# PR
precision_svm, recall_svm, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_svm = metrics.auc(recall_svm, precision_svm)

# ROC
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_true, y_scores)
auroc_svm = metrics.roc_auc_score(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_svm, pr_auc_svm))


# ## Density Forest

# In[ ]:


# Create DensityForest instance
clf_df = DensityForest(max_depth=3, min_subset=0, n_trees=20,
                       subsample_pct=.0001, n_jobs=-1, verbose=10, batch_size=10000)


# In[ ]:


clf_df.fit(act_val[pred_t_val.flatten()])


# In[ ]:


probas_df = clf_df.predict(act_test)


# In[ ]:


# metrics
y_scores = -probas_df

# PR
precision_df, recall_df, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_df = metrics.auc(recall_df, precision_df)

# ROC
fpr_df, tpr_df, _ = metrics.roc_curve(y_true, y_scores)
auroc_df = metrics.roc_auc_score(y_true, y_scores)

print("AUROC: %.2f, PR AUC: %.2f" % (auroc_df, pr_auc_df))


# # Plots

# In[ ]:


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
plt.savefig("Figures/PR_pred_wo_cl_" + str(class_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)
plt.close()


# In[ ]:


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
    plt.step(fprs[i], tprs[i], where='post', c=colors_lines[i])

plt.plot([0, 1], [0, 1], '--', c='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.legend([str.format('%s: %.2f') % (names_methods[i], scores_auc[i]) for i in scores_order], title="AUROC")
plt.savefig("Figures/ROC_pred_wo_cl_" + str(class_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)
plt.close()

