# To Do
- Implement **cross-validation** for Zurich dataset

### Density Forests
- Why does it work significantly better with lower dimensions?
- Why n_clusters not behaving expectedly as a function of tree detph, n_subsample?
- Subsample all classes equally for training DF?

### Hyperparameters to Test
- Density Forests: 
  - Max Depth
  - Improvement Factor
  - N_trees
- SVM
  - kernel (rbf or poly, degrees 3-9)
- GMM
  - n_clusters (3-15)
  

### Experiments
**Error Detection**: all classes, MNIST and Zurich dataset: detect wrongly predicted classes
- *Network*: MSR, Margin, Entropy
- *MC-Dropout*
- *Image Transformations*: Train MLP on transformed softmax activations of training set
- Activations: 
  - *Distance*: distance to *k* nearest neighbors in activations 
  - *One-class SVM* trained on correctly predicted labels of training set
  - *GMM* trained on correctly predicted labels of training set
  - *Density Forest* trained on activations of correct labels in training set

**Novelty Detection**: leaving out one class, MNIST and Zurich dataset: detect unseen class
- *Network*: MSR, Margin, Entropy
- *MC-Dropout*
- *Image Transformations*: Train MLP on transformed softmax activations of training set
- Activations: 
  - *One-class SVM* trained on activations of seen points in training set
  - *GMM* trained on activations of seen points in training set
  - *Density Forest* trained on activations of seen points in training set