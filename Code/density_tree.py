
# coding: utf-8

# # Density Tree for N-dimensional data and labels
# The code below implements a **density** tree for non-labelled data.

# ## Libraries
# First, some libraries are loaded and global figure settings are made for exporting.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm
from IPython.core.display import Image, display

# Custom Libraries
from density_tree.density_forest import *
from density_tree.density_tree_create import *
from density_tree.density_tree_traverse import *
from density_tree.create_data import *
from density_tree.helpers import *
from density_tree.plots import *


# # Generate Data
# 
# First, let's generate some unlabelled data:
# 

# In[2]:


dataset = 2
dimensions = 2

def fermat(n_points=500, divergence=0.05, n_spirals=5):
    theta = np.linspace(np.pi / 50, 7 * np.pi / 16, n_points)
    a = np.random.uniform(1 - divergence, 1 + divergence, n_points)

    x = a * np.sqrt(theta) * np.cos(theta)
    y = a * np.sqrt(theta) * np.sin(theta)

    n_spirals = 4
    for i in range(1, n_spirals):
        x_, y_ = rotate([0, 0], [x, y], i * 2 * np.pi / n_spirals)
        x = np.concatenate((x, x_))
        y = np.concatenate((y, y_))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.scatter(x, y, marker='.')
    plt.savefig("../Figures/density_tree/unlabelled-data-spiral.pdf", bbox_inches='tight')

    # save as new dataset
    dataset = np.asarray([x, y]).T
    minRange = np.min(x)
    maxRange = np.max(x)
    return dataset, minRange, maxRange

# Dataset 1: Random Points
if dataset == 1:
    nclusters = 4
    covariance = 2
    npoints = 1200
    minRange = 0
    maxRange = 100

    dataset = create_data(nclusters, dimensions, covariance, npoints, minrange=minRange, maxrange=maxRange,
                          labelled=False, random_flip=True, nonlinearities=True)

    if dimensions == 2:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8,6)
        plot_data(dataset, "Unlabelled data", ax, labels=False, minrange=minRange, maxrange=maxRange, margin=30)
        plt.savefig("../Figures/density_tree/unlabelled-data.pdf", bbox_inches='tight', pad_inches=0)

        plt.show()

# Dataset 2: Fermat Spiral

elif dataset==2:
    print("Creating data")
    dataset, minRange, maxRange = fermat()
    print("Done creating data")
    
# Dataset 3: S-shape
elif dataset==3:
    n_points = 300
    theta = np.linspace(np.pi/4,6*np.pi/4,n_points)
    divergence = 0.05
    a = np.random.uniform(1-divergence,1+divergence,n_points)

    x = a*np.sqrt(theta)*np.sin(theta)
    y = a*np.sqrt(theta)*np.cos(theta)*x


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.scatter(x,y, marker='.')
    plt.savefig("../Figures/density_tree/unlabelled-data-spiral.pdf", bbox_inches='tight')
    
    # save as new dataset
    dataset = np.asarray([x,y]).T
    minRange = np.min([np.min(x),np.min(y)])
    maxRange = np.max([np.max(x),np.max(y)])


# In[3]:


from ipywidgets import interact, FloatSlider
def fig(begin, end):
    ind = np.arange(begin,end)
    plt.scatter(dataset[ind,0],dataset[ind,1])
    plt.title("Entropy: %.3f" % entropy_gaussian(dataset[ind]))


min_ = 0
max_ = 5000
step_= 20
# max_ = len(dataset)
interact(fig,begin=(min_-step_,max_,step_),end=(min_,max_,step_))


# #### Create single Density Tree

# In[4]:


def get_values_preorder(node, cut_dims, cut_vals):
    cut_dims.append(node.split_dimension)
    cut_vals.append(node.split_value)
    if node.left is not None:
        get_values_preorder(node.left, cut_dims, cut_vals)
    if node.right is not None:
        get_values_preorder(node.right, cut_dims, cut_vals)
    return cut_vals, cut_dims

def plot_clusters_node(node, side):
    if side == 'l' or side is None:
        subset_dataset = node.get_dataset('l',dataset)
        plt.scatter(subset_dataset[:,0],subset_dataset[:,1])
    if side == 'r' or side is None:
        subset_dataset = node.get_dataset('r',dataset)
        plt.scatter(subset_dataset[:,0],subset_dataset[:,1])


    if node.parent is not None:
        if node.parent.left == node:
            plot_clusters_node(node.parent, 'r')
        else:
            plot_clusters_node(node.parent, 'l')
            
def plot_splits(dataset, root):
    cut_vals, cut_dims = get_values_preorder(root, [], [])
    cut_vals = np.asarray(cut_vals).astype(float)
    cut_dims = np.asarray(cut_dims).astype(int)

    # show splits
    x_split = cut_vals[cut_dims == 0]
    y_split = cut_vals[cut_dims == 1]
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    plot_data(dataset, "Training data after splitting", ax, labels=False, lines_x=x_split, lines_y=y_split,
              minrange=minRange, maxrange=maxRange, margin=.3)

    plt.axis('off')
    plt.show()

    
def plot_ellipses(dataset, root, covs, means, ax, subsample_pct=.05):
    plot_data(draw_subsamples(dataset, subsample_pct), "Unlabelled data", ax, labels=False, n_clusters=5, covs=covs, means=means,
              minrange = minRange, maxrange = maxRange, margin=.3)
    #plt.savefig("../Figures/density_tree/unlabelled-data-cov.pdf", bbox_inches='tight', pad_inches=5)
    #plt.axis('off')


# In[5]:


#warnings.filterwarnings("ignore")
max_depth = 4
fact_improvement = 0
min_subset = 0.001
n_dim_max = 0

print("Dataset size: %i" % int(len(dataset)))
print("Minimum number of points in each leaf: %i" % int(len(dataset) * min_subset))

root = create_density_tree(dataset, max_depth=max_depth, min_subset=min_subset, n_max_dim=n_dim_max, 
                                verbose=False, fact_improvement=fact_improvement)

covs, means = get_clusters(root, [], [])    
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8,8)
plot_ellipses(dataset, root, covs, means, ax, subsample_pct=.005)
plt.show()


# In[ ]:


def get_covs_means_depth(node, depth, means, covs):
    """get covs and means of a tree at a certain depth"""
    if node.depth() == depth:
        means.append(node.mean)
        covs.append(node.cov)
    if node.depth() <= depth and node.right is None:
        means.append(node.right_mean)
        covs.append(node.right_cov)

    if node.depth() <= depth and node.left is None:
        means.append(node.left_mean)
        covs.append(node.left_cov)
        
    if node.left is not None:
        means, covs =  get_covs_means_depth(node.left, depth, means, covs)

    if node.right is not None:
        means, covs = get_covs_means_depth(node.right, depth, means, covs)
    
    return means, covs    


# In[ ]:


fig, axes = plt.subplots(2, 4)
fig.set_size_inches(15,8)

for i in range(1,5):
    means, covs = get_covs_means_depth(root,i,[],[])
    plot_ellipses(dataset, root, covs, means, axes[0][i-1])
    
for i in range(5,9):
    means, covs = get_covs_means_depth(root,i,[],[])
    plot_ellipses(dataset, root, covs, means, axes[1][i-5])


# # Printing the Tree

# In[ ]:


# ATTENTION: requires LaTeX, imagemagick to be installed

def tree_visualize(root):
    tree_string = ""
    tree_string = print_density_tree_latex(root, tree_string)
    cdir = "cd ../Figures/density_tree;"
    
    os.system(cdir + "rm density-tree.tex")
    os.system(cdir + "more main_pt1.tex >> density-tree.tex; echo '' >> density-tree.tex;")
    os.system(cdir + "echo '" + tree_string + "' >> density-tree.tex;  more main_pt2.tex >> density-tree.tex;")
    os.system(cdir + "/Library/TeX/texbin/pdflatex density-tree.tex")
    os.system(cdir + "convert -density 300 -trim density-tree.pdf -quality 100 density-tree.png")
    os.system(cdir + "rm *.aux *.log")
    display(Image('../Figures/density_tree/density-tree.png', retina=True))
    
tree_visualize(root)


# In[ ]:


from scipy.stats import multivariate_normal

# loop
def create_plot_surface(minRange, maxRange, roots, n_points_dim=100, standardize=False, thresh=.01):
    """thresh: pct of data in each leaf node"""
    grid = np.linspace(minRange, maxRange, n_points_dim)
    probas_out = np.zeros((n_points_dim,n_points_dim))

    
    for idx_x, pt in tqdm(enumerate(grid)):
        for idx_y, g in enumerate(grid):
            proba_pt=[]
            for root in roots:
                # descend tree
                d_mean, d_pct, d_pdf_mean, d_cov_det, d_cov_inv = descend_density_tree([pt, g], root)
                # probability for this point to be from this distribution
                if d_pct>thresh:
                    try:
                        mn = my_normal(np.asarray([pt, g]), d_mean, d_cov_det, d_cov_inv)
                        if standardize: 
                            mn /= d_pdf_mean
                        proba_pt.append(mn)
                    except:
                        proba_pt.append(np.nan)

                
            probas_out[idx_x, idx_y] = np.nanmean(proba_pt)

    return probas_out
    fig = plt.figure(figsize=(7,7))
    plt.imshow(probas_out)


# In[ ]:


probas = create_plot_surface(minRange, maxRange, [root], n_points_dim=100, standardize=False, thresh=0)
fig = plt.figure(figsize=(7,7))
plt.imshow(probas)
plt.axis('off')
plt.show()
#plt.savefig("../Figures/density_tree/spirals_one_tree_maxDepth"+str(max_depth)+".pdf", bbox_inches='tight', pad_inches=0)


# #### Density Forest

# In[ ]:


n_trees = 20
subsample_pct = .1
min_subset = .2
n_jobs = -1
n_max_dim = -1

print("Number of points on which to train each tree: %i" % int(len(dataset) * subsample_pct))
print("Minimum number of points in each leaf: %i" % int(len(dataset) * subsample_pct * min_subset))

root_nodes = density_forest_create(dataset, max_depth=max_depth, min_subset=min_subset, 
                                   n_trees=n_trees, subsample_pct=subsample_pct, n_max_dim=n_max_dim, 
                                   n_jobs=n_jobs, fact_improvement=fact_improvement)


# In[ ]:


root_nodes = np.asarray(root_nodes)
root_nodes = root_nodes[np.not_equal(root_nodes, None)]


# In[ ]:


probas = create_plot_surface(minRange, maxRange, root_nodes, n_points_dim=100, standardize=False, thresh=0.001)

fig = plt.figure(figsize=(7,7))
plt.imshow(probas) 
plt.axis('off')
plt.savefig("../Figures/density_tree/spirals_forest_maxDepth"+str(max_depth)+"_nTrees"+str(n_trees)+"_pct"+str(subsample_pct)+".pdf", bbox_inches='tight', pad_inches=0)
plt.colorbar()


# In[ ]:


# standardized version
probas = create_plot_surface(minRange, maxRange, root_nodes, n_points_dim=100, standardize=True, thresh=.001)

fig = plt.figure(figsize=(7,7))
plt.imshow(probas)
plt.axis('off')
plt.savefig("../Figures/density_tree/spirals_forest_maxDepth"+str(max_depth)+"_nTrees"+str(n_trees)+"_pct"+str(subsample_pct)+".pdf", bbox_inches='tight', pad_inches=0)
plt.colorbar()


# In[ ]:


def split_x(dataset, index, split_value):
    """
    split a dataset (columns: variables, rows: data) in two according to some column (index) value.
    :param dataset: input dataset
    :param index: index of dimension to split values on
    :param split_value: value of the dimension where the dataset is split
    :return: left and right split datasets
    """

    
    return left, right


# In[ ]:


dataset[:,1] < .1

