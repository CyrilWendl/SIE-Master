
# coding: utf-8

# # MNIST Dataset: Density Forests
# Trains a simple CNN on _N-1_ classes of the MNIST dataset, predicts unseen classes using baseline methods and Density Forest.

# In[1]:


# choose GPUs
import os, sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

# In[2]:


#import libraries
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils as np_utils
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn import decomposition, svm
from tensorflow.python.client import device_lib
import pandas as pd

#custom libraries
#base_dir = '/Users/cyrilwendl/Documents/EPFL'
base_dir = '/raid/home/cwendl'  # for guanabana
import sys
sys.path.append(base_dir + '/SIE-Master/Code')  # Path to density Tree package
from density_forest.density_forest import *
from density_forest.helpers import *
from baselines.helpers import *
from helpers.helpers import *
from helpers.plots import *
from helpers.parameter_search import *

# print available devices
print(device_lib.list_local_devices())

# # Data Import 
# Import the data, delete all data in the training set of class 7
# 

# In[1]:


df_ps = pd.read_csv('mnist_models/hyperparams.csv', index_col=0)
df_ps

# In[3]:


label_to_remove = int(sys.argv[1])
paramsearch=True

# the data, shuffled and split between train and test sets
def load_data(label_to_remove):
    (x_train_all, y_train_all), (x_test_all, y_test_all) = mnist.load_data()
    print(np.unique(y_train_all, return_counts=True))

    # remove all trainig samples containing a label label_to_remove
    x_train = x_train_all[y_train_all != label_to_remove]
    y_train = y_train_all[y_train_all != label_to_remove]

    x_test = x_test_all[y_test_all != label_to_remove]
    y_test = y_test_all[y_test_all != label_to_remove]

    # decrease all labels that are higher by -1 to avoid gaps
    for i in range(label_to_remove + 1, 11):
        y_train[y_train == i] = (i-1)
        y_test[y_test == i] = (i-1)
    print(np.unique(y_train, return_counts=True))    
    
    batch_size = 128
    num_classes = 9
    epochs = 5

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Reshape for Tensorflow
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test_all = x_test_all.reshape(x_test_all.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_test_all = x_test_all.astype('float32')
    x_train /= 255
    x_test /= 255
    x_test_all /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.np_utils.to_categorical(y_test, num_classes)


    return x_train_all, y_train_all, x_test_all, y_test_all, x_train, y_train, x_test, y_test, batch_size, num_classes, epochs, img_rows, img_cols, input_shape

x_train_all, y_train_all, x_test_all, y_test_all, x_train, y_train, x_test, y_test, batch_size, num_classes, epochs, img_ros, img_cols, input_shape = load_data(label_to_remove)

# In[4]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())

model_train = False
if model_train:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_test, y_test))
    model.save('mnist_models/mnist-weights-' + str(label_to_remove) + '.h5')
else:
    model = load_model('mnist_models/mnist-weights-' + str(label_to_remove) + '.h5')

# In[5]:


def OA(y_true, y_pred):
        """get overall accuracy"""
        return np.sum(y_true == y_pred)/len(y_true)

def AA(y_true, y_pred):
    """get average (macro) accuracy"""
    acc_cl = []
    for label in np.unique(y_true):
        acc_cl.append(np.sum(y_true[y_pred==label] == y_pred[y_pred==label])/len(y_pred[y_pred==label]))
    return np.mean(acc_cl), acc_cl

# In[6]:


# get all predictions in training and test set
oa_trs = []
aa_trs = []
oa_tes = []
aa_tes = []

# for label_to_remove in range(10):
"""
get mean oa, aa for tr and te for all labels
"""
print(label_to_remove)
x_train_all, y_train_all, x_test_all, y_test_all, x_train, y_train, x_test, y_test, batch_size, num_classes, epochs, img_ros, img_cols, input_shape = load_data(label_to_remove)
model = load_model('mnist_models/mnist-weights-' + str(label_to_remove) + '.h5')
y_pred_tr = model.predict(x_train_all[..., np.newaxis], verbose=True)
y_pred_label_tr = get_y_pred_labels(y_pred_tr, class_to_remove=label_to_remove, background=False)

y_pred_te = model.predict(x_test_all, verbose=True)
y_pred_label_te = get_y_pred_labels(y_pred_te, class_to_remove=label_to_remove, background=False)

# get indices of correctly / incorrectly predicted images
pred_t_tr = y_train_all != label_to_remove
pred_f_tr = y_train_all == label_to_remove

pred_t_te = y_test_all != label_to_remove
pred_f_te = y_test_all == label_to_remove

# get overall and average accuracy for training and test set


oa_tr = OA(y_train_all[y_train_all!=label_to_remove], y_pred_label_tr[y_train_all!=label_to_remove])
aa_tr, aa_tr_cl = AA(y_train_all[y_train_all!=label_to_remove], y_pred_label_tr[y_train_all!=label_to_remove])

oa_te = OA(y_test_all[y_test_all!=label_to_remove], y_pred_label_te[y_test_all!=label_to_remove])
aa_te, aa_te_cl = AA(y_test_all[y_test_all!=label_to_remove], y_pred_label_te[y_test_all!=label_to_remove])
oa_trs.append(oa_tr)
aa_trs.append(aa_tr)

oa_tes.append(oa_te)
aa_tes.append(aa_te)

print("OA training set: %.2f" % (oa_tr*100))
print("AA training set: %.2f" % (aa_tr*100))

print("OA test set: %.2f" % (oa_te*100))
print("AA test set: %.2f" % (aa_te*100))

# In[7]:


print(np.round(np.multiply(oa_trs,100),2))
print(np.round(np.multiply(aa_trs,100),2))
print(np.round(np.multiply(oa_tes,100),2))
print(np.round(np.multiply(aa_tes,100),2))

# #### Make some predictions for the unseen class

# In[8]:


# all images in the test set containing a label label_to_remove
x_unseen_class = x_test_all[np.where(y_test_all == label_to_remove)[0]] 

# make prodictions for class unseen during training
y_pred = model.predict(x_unseen_class)
y_pred_label = get_y_pred_labels(y_pred, label_to_remove, background=False)

# distribution of predicted label
pred_labels, pred_counts = np.unique(y_pred_label, return_counts=True)

# visualization
fig = plt.figure(figsize=(7, 5))
plt.bar(pred_labels, pred_counts)
plt.xticks(np.arange(0, 10))
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.title("Misclassified labels (mean MSR=%.2f)" % np.mean(get_acc_net_msr(y_pred)))
plt.savefig("../Figures/MNIST/Pred_count/pred-count_wo_cl" + str(label_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

# In[9]:


# Avarage certitude for unseen class: 1-max_margin
c = get_acc_net_max_margin(y_pred)
    
pred_acc_mean = np.mean(c)
pred_acc_std = np.std(c)
    
print("Mean accuracy: %.2f %%" % (pred_acc_mean * 100))
print("Std accuracy: %.2f %%" % (pred_acc_std * 100))

pred_acc_high = .95

pct = np.round(len(c[c > pred_acc_high]) / len(c), 4) * 100
print("%.2f%% of all predictions made with an accuracy higher than %.2f%%" % (pct, pred_acc_high))

# In[10]:


# Avarage certitude for seen class: 1-max_margin
y_pred_seen = model.predict(x_test)
y_pred_label_seen = get_y_pred_labels(y_pred_seen, label_to_remove, background=False)

c = get_acc_net_max_margin(y_pred_seen)
    
pred_acc_mean = np.mean(c)
pred_acc_std = np.std(c)
    
print("Mean accuracy: %.2f %%" % (pred_acc_mean * 100))
print("Std accuracy: %.2f %%" % (pred_acc_std * 100))

pred_acc_high = .95

pct = np.round(len(c[c > pred_acc_high])/len(c), 4) * 100
print("%.2f %% of all predictions made with an accuracy higher than %.2f%%" % (pct, pred_acc_high))

# ## Network accuracy

# In[11]:


# precision-recall curves

# msr
y_scores = 1-get_acc_net_msr(y_pred_te)
y_true = pred_f_te
precision_msr, recall_msr, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_msr = metrics.average_precision_score(y_true, y_scores)
auroc_msr = metrics.roc_auc_score(y_true, y_scores)
fpr_msr, tpr_msr, _ = metrics.roc_curve(y_true, y_scores)

# margin
y_scores = 1-get_acc_net_max_margin(y_pred_te)
precision_margin, recall_margin, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_margin = metrics.average_precision_score(y_true, y_scores)
auroc_margin = metrics.roc_auc_score(y_true, y_scores)
fpr_margin, tpr_margin, _ = metrics.roc_curve(y_true, y_scores)

# entropy
y_scores = 1-get_acc_net_entropy(y_pred_te)
precision_entropy, recall_entropy, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_entropy = metrics.average_precision_score(y_true, y_scores)
auroc_entropy = metrics.roc_auc_score(y_true, y_scores)
fpr_entropy, tpr_entropy, _ = metrics.roc_curve(y_true, y_scores)

# plot
plt.step(recall_msr, precision_msr, where='post')
plt.step(recall_margin, precision_margin, where='post')
plt.step(recall_entropy, precision_entropy, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend([str.format('MSR (AUC=%.2f)' % pr_auc_msr),
            str.format('Margin (AUC=%.2f)' % pr_auc_margin),
            str.format('Entropy (AUC=%.2f)' % pr_auc_entropy)])

# # Dropout

# In[12]:


y_preds = predict_with_dropouts_batch(model, x_test_all, 
                                      batch_size=100, n_iter=5)

# In[13]:


y_pred = np.mean(y_preds, axis=0)
probas = -get_acc_net_entropy(y_pred)

# In[14]:


# Metrics
# PR
precision_dropout, recall_dropout, _ = metrics.precision_recall_curve(y_true, probas)
pr_auc_dropout = metrics.auc(recall_dropout, precision_dropout)
# ROC
fpr_dropout, tpr_dropout, _ = metrics.roc_curve(y_true, probas)
auroc_dropout = metrics.roc_auc_score(y_true, probas)

# # Density Forest
# ## Get Activations, PCA, t-SNE

# In[15]:


# get activation weights of last layer
act_unseen = get_activations_batch(model, 6, x_unseen_class, verbose=True)
act_train_all = get_activations_batch(model, 6, x_train_all[..., np.newaxis], verbose=True)
act_train = act_train_all[y_train_all != label_to_remove]
act_test = get_activations_batch(model, 6, x_test_all, verbose=True)

# In[16]:


pts_per_class = 150
n_classes = 10
dataset_subset_indices = get_balanced_subset_indices(y_test_all, np.arange(n_classes), pts_per_class)

# In[17]:


tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
# t-SNE visualization before PCA
tsne_all = tsne.fit_transform(act_test[np.concatenate(dataset_subset_indices)])

# In[18]:


# color scale and legend for t-sne plots
colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))[:, :3]
names = ['Class ' + str(i) for i in range(10)]
classes_to_keep = np.asarray([x for x in range(n_classes) if x != label_to_remove])

# plot
tsne_y = y_test_all[np.concatenate(dataset_subset_indices)]

# In[20]:


# plot
_, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_pts_2d(tsne_all, tsne_y, ax, classes_to_keep,
            colors, class_to_remove=label_to_remove)
ax.set_axis_off()

plt.savefig("../Figures/MNIST/tSNE/MNIST_t-SNE_wo_cl_" + str(label_to_remove) + "_before.pdf", bbox_inches='tight', pad_inches=0)

# In[21]:


pca = decomposition.PCA(n_components=.95)
pca.fit(act_train)

# In[22]:


# PCA
pca = decomposition.PCA(n_components=.95)
pca.fit(act_train)


# plot
print("Variance explained by first %i components: %.2f" % (len(pca.components_), 
                                                           np.sum(pca.explained_variance_ratio_[:len(pca.components_)])))

fig = plt.figure(figsize=(7, 5))
plt.scatter(np.arange(len(pca.components_)), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative sum of explained variance")
plt.grid(alpha=.3)
fig.axes[0].spines['right'].set_visible(False)
fig.axes[0].spines['top'].set_visible(False)
plt.savefig("../Figures/MNIST/PCA/MNIST_pca_components_wo_cl_" + str(label_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

# In[23]:


# fit PCA
X_train_all = pca.transform(act_train_all)
X_train = pca.transform(act_train)
X_test = pca.transform(act_test)

# #### Visualize PCA

# In[24]:


# test sample (with unseen class)
plot_pts_3d(X_train_all, y_train_all, classes_to_keep, colors, class_to_remove=label_to_remove, 
            subsample_pct=.05, s_name="../Figures/MNIST/PCA/MNIST_PCA_3D_wo_cl_" + str(label_to_remove) + ".pdf")

# In[25]:


# test sample (with unseen class)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_pts_2d(X_train_all, y_train_all, ax, classes_to_keep, colors, class_to_remove=label_to_remove, 
            subsample_pct=.05)
ax.set_axis_off()
plt.savefig("../Figures/MNIST/PCA/MNIST_PCA_2D_wo_cl_" + str(label_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

# In[26]:


# t-SNE visualization after PCA
tsne_all = tsne.fit_transform(X_test)

# In[27]:


# plot
_, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_pts_2d(tsne_all, y_test_all, ax, classes_to_keep,
            colors, class_to_remove=label_to_remove)
ax.set_axis_off()

plt.savefig("../Figures/MNIST/tSNE/MNIST_t-SNE_wo_cl_" + str(label_to_remove) + "_after.pdf", bbox_inches='tight', pad_inches=0)

# In[28]:


tsne_train = tsne_all[y_test_all != label_to_remove]

# ### GMM
# GMM, calculate 

# In[78]:


# parameter search
if paramsearch:
    tuned_parameters = [{'n_components': np.arange(3, 10), 'max_iter': [10000]}]
    # do parameter search
    ps_gmm = ParameterSearch(GaussianMixture, tuned_parameters, X_train, X_train_all,
                             pred_f_tr.flatten(), scorer_roc_probas_gmm, 
                             n_iter=3, verbosity=10, n_jobs=-1, subsample_train=.01, subsample_test=.01)
    ps_gmm.fit()
    best_params_gmm = ps_gmm.best_params

else:
    best_params_gmm = {'n_components': df_ps.loc[str(names[label_to_remove])]['gmm_n_components'], 'max_iter': 10000}

print(best_params_gmm)

# In[31]:


# fit model
gmm = GaussianMixture(**best_params_gmm)
gmm.fit(X_test)

# In[32]:


# predict
probas = gmm.score_samples(X_test)

# In[33]:


# Metrics
# PR
precision_gmm, recall_gmm, _ = metrics.precision_recall_curve(y_true, -probas)
pr_auc_gmm = metrics.auc(recall_gmm, precision_gmm)
# ROC
fpr_gmm, tpr_gmm, _ = metrics.roc_curve(y_true, -probas)
auroc_gmm = metrics.roc_auc_score(y_true, -probas)

print("PR AUC: %.2f, AUROC: %.2f" % (pr_auc_gmm, auroc_gmm))

# ## One-Class SVM

# In[34]:


X_train_svm = preprocessing.scale(X_train)
X_train_all_svm = preprocessing.scale(X_train_all)
X_test_svm = preprocessing.scale(X_test)

# In[38]:


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
    ps_svm = ParameterSearch(svm.OneClassSVM, tuned_parameters, X_train_svm, X_train_all_svm,
                             pred_f_tr.flatten(), scorer_roc_probas_svm, n_iter=5,
                             verbosity=11, n_jobs=-1, subsample_train=.001, subsample_test=.01)
    ps_svm.fit()
    best_params_svm = ps_svm.best_params
else:
    best_params_svm = {'kernel': df_ps.loc[str(names[label_to_remove])]['oc-svm_k'], 
                       'degree': df_ps.loc[str(names[label_to_remove])]['oc-svm_deg'], 
                       'nu': df_ps.loc[str(names[label_to_remove])]['oc_svm_nu'],
                       'max_iter': 100}
    
print(best_params_svm)

# In[39]:


clf_svm = svm.OneClassSVM(**best_params_svm, verbose=True)

# In[40]:


clf_svm.fit(X_train_svm)

# In[41]:


probas = clf_svm.decision_function(X_test_svm)[:, 0]
probas -= np.min(probas)
probas /= np.max(probas)

# In[42]:


# Metrics
# PR
precision_svm, recall_svm, _ = metrics.precision_recall_curve(y_true, -probas)
pr_auc_svm = metrics.auc(recall_svm, precision_svm)
# ROC
fpr_svm, tpr_svm, _ = metrics.roc_curve(y_true, -probas)
auroc_svm = metrics.roc_auc_score(y_true, -probas)

print("PR AUC: %.2f, AUROC: %.2f" % (pr_auc_svm, auroc_svm))

# # Density Forest

# In[54]:


# parameter search
default_params = {'n_trees': 10, 'n_max_dim': 0, 'n_jobs': -1,
                  'verbose': 0, 'subsample_pct': .05, 'min_subset': 0}

if paramsearch:
    """search for best hyperparameters"""
    tuned_params = [{'max_depth': [1, 2, 3],
                     'ig_improvement': [0, .4, .7, .9]
                    }]

    # do parameter search
    ps_df = ParameterSearch(DensityForest, tuned_params, X_train, X_train_all,
                            pred_f_tr.flatten(), scorer_roc_probas_df,
                            n_iter=3, verbosity=11, n_jobs=1, subsample_train=1, 
                            subsample_test=.1, default_params=default_params)

    print("Testing %i combinations %i times" % (len(ps_df.combinations), ps_df.n_iter))
    print(ps_df.combinations)
    ps_df.fit()
    print(ps_df.best_params)
    
    # Create DensityForest instance
    best_params_df = ps_df.best_params
    
else:
    """use previously found hyperparameters"""
    best_params_df = {'max_depth': df_ps.loc[str(names[label_to_remove])]['df_depth'],
                      'ig_improvement': df_ps.loc[str(names[label_to_remove])]['df_min_ig']}
    
    
print(best_params_df)
default_params['verbose'] = 1
default_params['batch_size'] = 100
default_params['n_trees'] = 20

# In[65]:


# Create DensityForest instance
clf_df = DensityForest(**best_params_df, **default_params)
clf_df.fit(X_train)

# In[67]:


probas = clf_df.predict(X_test)

# In[68]:


covs, means = get_clusters(clf_df.root_nodes[1], [], [])

# In[69]:


# precision-recall curve
y_scores = -probas
# PR
precision_df, recall_df, _ = metrics.precision_recall_curve(y_true, y_scores)
pr_auc_df = metrics.auc(recall_df, precision_df)
# ROC
fpr_df, tpr_df, _ = metrics.roc_curve(y_true, y_scores)
auroc_df = metrics.roc_auc_score(y_true, y_scores)
plt.step(recall_df, precision_df)
metrics.auc(recall_df, precision_df)

# ## Plot Results

# In[70]:


# Precision-Recall Curve
# order according to increasing score
scores_pr = np.multiply([pr_auc_msr, pr_auc_margin, pr_auc_entropy, 
                         pr_auc_dropout, pr_auc_gmm, pr_auc_svm, pr_auc_df], 100)

recalls = [recall_msr, recall_margin, recall_entropy, recall_dropout, recall_gmm, recall_svm, recall_df]
precisions = [precision_msr, precision_margin, precision_entropy, 
              precision_dropout, precision_gmm, precision_svm, precision_df]

names_methods = np.array(['MSR', 'Margin', 'Entropy', 'Dropout', 'GMM', 'OC SVM', 'DF'])
scores_order = np.argsort(scores_pr)
colors_lines = plt.cm.rainbow(np.linspace(0, 1, len(scores_pr)))[:, :3]

# plot
plt.figure(figsize=(6, 6))
for i in scores_order:
    plt.step(recalls[i], precisions[i], where='post', c=colors_lines[i])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend([str.format('%s: %.2f') % (names_methods[i], scores_pr[i]) for i in scores_order], title="PR AUC [%]")
plt.savefig("../Figures/MNIST/Metrics/PR_ED_wo_cl_" + str(label_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

# In[71]:


# ROC
# order according to increasing score
scores_auc = np.multiply([auroc_msr, auroc_margin, auroc_entropy, auroc_dropout, auroc_gmm, auroc_svm, auroc_df], 100)
fprs = [fpr_msr, fpr_margin, fpr_entropy, fpr_dropout, fpr_gmm, fpr_svm, fpr_df]
tprs = [tpr_msr, tpr_margin, tpr_entropy, tpr_dropout, tpr_gmm, tpr_svm, tpr_df]
scores_order = np.argsort(scores_auc)
colors_lines = plt.cm.rainbow(np.linspace(0, 1, len(scores_auc)))[:, :3]

# plot
plt.figure(figsize=(6, 6))
for i in scores_order:
    plt.step(fprs[i], tprs[i], where='post', c=colors_lines[i])

plt.plot([0, 1], [0, 1], '--', c='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend([str.format('%s: %.2f') % (names_methods[i], scores_auc[i]) for i in scores_order], title="AUROC [%]")
plt.savefig("../Figures/MNIST/Metrics/ROC_ED_wo_cl_" + str(label_to_remove) + ".pdf", bbox_inches='tight', pad_inches=0)

# In[79]:


# write best hyperparameters to CSV file
best_params = {'gmm_n_components': best_params_gmm['n_components'], 
               'oc-svm_k': best_params_svm['kernel'], 
               'oc-svm_deg': best_params_svm['degree'], 
               'oc_svm_nu': best_params_svm['nu'], 
               'df_depth': best_params_df['max_depth'], 
               'df_min_ig': best_params_df['ig_improvement']}

# In[89]:


# write results to CSV files
# hyperparameters
df2 = pd.DataFrame(best_params, index=[str(names[label_to_remove])])
df_ps = df_ps.append(df2)
df_ps = df_ps[~df_ps.index.duplicated(keep='last')]  # avoid duplicates
df_ps.to_csv('mnist_models/hyperparams.csv')

# AUROC
df_auroc = pd.read_csv('mnist_models/auroc_all.csv', index_col=0)
df2 = pd.DataFrame({str(names[label_to_remove]): scores_auc}, index=names_methods).T
df_auroc = df_auroc.append(df2)
df_auroc = df_auroc[~df_auroc.index.duplicated(keep='last')]  # avoid duplicates
df_auroc.to_csv('mnist_models/auroc_all.csv')


# PR AUC
df_aucpr = pd.read_csv('mnist_models/aucpr_all.csv', index_col=0)
df2 = pd.DataFrame({str(names[label_to_remove]): scores_pr}, index=names_methods).T
df_aucpr = df_aucpr.append(df2)
df_aucpr = df_aucpr[~df_aucpr.index.duplicated(keep='last')]  # avoid duplicates
df_aucpr.to_csv('mnist_models/aucpr_all.csv')
