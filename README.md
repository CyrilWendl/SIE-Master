# Density Forest 
Code Repository of the EPFL SIE Master Project, Spring Semester 2018.

The goal of this project is to perform error detection and novelty detection in Convolutional Neural Networks (CNNs) using Density Forests. Applications to the MNIST dataset and a dataset for semantic segmentation of land cover classes in Zurich are visualized in  `Code/` and `Zurich/`.

## 📈 Visualization
Density trees maximize Gaussianity at each split level. In 2D this might look as follows:

![Simple 2D visualization](Figures/density_tree/gif/splits_visu.gif) 

A density forest is a collection of density trees each trained on a random subset of all data.

![t-SNE of pre-softmax activations of Zurich dataset](Figures/Zurich/GIF/tsne_act.gif) 

The above example shows the t-SNE of the pre-softmax activations of a network trained for semantic segmentation of the
 Zurich dataset, leaving out one class during training. 
Density trees were trained on bootstrap samples of all classes but the unseen one. 

Confidence of each data point in the test set, the probability is calculated as the average Gaussian likelihood to come from the leaf node clusters.

![Probas](Figures/Zurich/GIF/probas.png)

Darker points represent regions of lower certainty and crosses represent activations of unseen classes.
 
 

## 📖 Usage of the `DensityForest` class:
#### Fitting a Density Forest
Suppose you have your training data `X_train` and test data `X_test`, in `[N, D]` with `N` data points in `D` dimensions:

```python
from density_forest.density_forest import DensityForest

clf_df = DensityForest(**params)  # create new class instance, put hyperparameters here
clf_df.fit(X_train)               # fit to a training set
conf = clf_df.predict(X_test)     # get confidence values for test set
```
Hyperparameters are documented in the docstring. To find the optimal hyperparameters, consider the section below.

#### Finding Hyperparameters
To find the optimal hyperparameters, use the `ParameterSearch` from `helpers.cross_validator`, which allows CV, and hyperparameter search.

```python
from helpers.cross_validator import ParameterSearch

# define hyperparameters to test
tuned_params = [{'max_depth':[2, 3, 4], 'n_trees': [10, 20]}] # optionally add non-default arguments as single-element arrays
default_params = [{'verbose':0, ...}]  # other default parameters 
# do parameter search
ps = ParameterSearch(DensityForest, tuned_parameters, X_train, X_train_all, y_true_tr, f_scorer, n_iter=2, verbosity=0, n_jobs=1, default_params=default_params)
ps.fit()

# get model with the best parameters, as above
clf_df = DensityForest(**ps.best_params, **default_params)  # create new class instance with best hyperparameters
...  # continue as above
```
Check the docstrings for more detailed documentation af the `ParameterSearch` class.


## 🗂 File Structure

### 👾 Code
All libraries for density forests, helper libraries for semantic segmentation and for baselines. 
#### `Code/density_forest/`
Package for implementation of Decision Trees, Random Forests, Density Trees and Density Forests
- `create_data.py`: functions for generating labelled and unlabelled data
- `decision_tree.py`: data structure for decision tree nodes
- `decision_tree_create.py`: functions for generating decision trees
- `decision_tree_traverse.py`: functions for traversing a decision tree and predicting labels
- `density_tree.py`: data struture for density tree nodes
- `density_tree_create.py`: functions for generating a density tree
- `density_tree_traverse.py`: functions for descending a density tree and retreiving its cluster parameters
- `density_forest.py`: functions for creating density forests
- `helper.py`: various helper functions
- `plots.py`: functions for plotting the data
- `random_forests.py`: functions for creating random forests

#### `Code/helpers`: 
General helpers library for semantic segmentation
- `data_augment.py`: custom data augmentation methods applied to both the image and the ground truth
- `data_loader.py`: PyTorch data loader for Zurich dataset
- `helpers.py`: functions for importing, cropping, padding images and other related image tranformations
- `parameter_search.py`: functions for finding optimal hyperparameters for Density Forest, OC-SVM and GMM (explained above)
- `plots.py`: Generic plotter functions for labelled 2D and 3D plots, used for t-SNE and PCA plots

#### `Zurich Land Cover/keras_helpers`
Helper functions for Keras
- `helpers.py`: get activations
- `callbacks.py`: callbacks to be evaluated after each epoch
- `unet.py`: UNET model for training of network on Zurich dataset

#### `Zurich Land Cover/baselines`

### 🗾 Visualizations
#### `Code/`: 
Visualizations of basic decision tree and density tree
- `decision_tree.ipynb`: Decision Trees and Random Forest on randomly generated labelled data
- `density_tree.ipynb`: Density Trees on randomly generated unlabelled data

#### `MNIST/`:
- `MNIST Novelty Detection.ipynb`: Training of a CNN leaving out one class, baselines and DF for novelty detection
- `MNIST Error Detection.ipynb`: Training of a CNN, baselines and DF for error detection

#### `Zurich/`
- `Zurich Dataset Novelty Detection.ipynb`: Training of CNN, baselines and DF for novelty detection
- `Zurich Dataset Error Detection.ipynb`: Training of CNN, baselines and DF for error detection

## 🎓 Supervisors:
- Prof. Devis Tuia, University of Wageningen
- Diego Marcos González, University of Wageningen
- Prof. François Golay, EPFL

Cyril Wendl, 2018
