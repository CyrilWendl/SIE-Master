# Density Forest 
This library was developed within an EPFL Master Project, Spring Semester 2018.

GitHub repository: https://github.com/CyrilWendl/SIE-Master
 

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
#### `density_forest/`
Package for implementation of Decision Trees, Random Forests, Density Trees and Density Forests
- `create_data.py`: functions for generating labelled and unlabelled data
- `decision_tree.py`: data structure for decision tree nodes
- `decision_tree_create.py`: functions for generating decision trees
- `decision_tree_traverse.py`: functions for traversing a decision tree and predicting labels
- `density_forest.py`: functions for creating density forests
- `density_tree.py`: data struture for density tree nodes
- `density_tree_create.py`: functions for generating a density tree
- `density_tree_traverse.py`: functions for descending a density tree and retrieving its cluster parameters
- `helper.py`: various helper functions
- `random_forests.py`: functions for creating random forests

#### `helpers/`: 
General helpers library for semantic segmentation
- `data_augment.py`: custom data augmentation methods applied to both the image and the ground truth
- `data_loader.py`: PyTorch data loader for Zurich dataset
- `helpers.py`: functions for importing, cropping, padding images and other related image tranformations
- `parameter_search.py`: functions for finding optimal hyperparameters for Density Forest, OC-SVM and GMM (explained above)
- `plots.py`:  Generic plotter functions for labelled and unlabelled 2D and 3D plots, used for t-SNE and PCA plots

#### `baselines/`:
Helper functions for confidence estimation baselines MSR, margin, entropy and MC-Dropout

#### `keras_helpers/`
Helper functions for Keras
- `helpers.py`: get activations
- `callbacks.py`: callbacks to be evaluated after each epoch
- `unet.py`: UNET model for training of network on Zurich dataset

### 🗾 Visualizations
#### `density_forest/`: 
Visualizations of basic decision tree and density tree
- `Decision Forest.ipynb`: Decision Trees and Random Forest on randomly generated labelled data
- `Density Forest.ipynb`: Density Trees on randomly generated unlabelled data

## 🎓 Supervisors:
- Prof. Devis Tuia, University of Wageningen
- Diego Marcos González, University of Wageningen
- Prof. François Golay, EPFL

Cyril Wendl, 2018
