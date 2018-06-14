# To Do

### Baseline Methods 
- Implement 3b - Confidence estimation in Neural Networks.pdf

### Ideas 
- Account for prediction variability
- Apply to active learning (seen classes)

### Density Forests
- Why does it work significantly better in less dimensions?
- Why n_clusters not behaving expectedly as a function of tree detph, n_subsample? 
- Subsample all classes equally for training DF?

### Hyperparameters to Test
- Density Forests: 
  - Max Depth
  - Improvement Factor
  - N_trees
- SVM
  - kernel (poly, rbf, degrees 3-9)
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

### Evaluation Metric
- AUC of recall-precision curve
- AUROC

#### MNIST: Novelty Detection PR AUC

|   Class |   0 |  1  |  2 |  3 | 4 |  5 | 6 |  7 | 8 | 9 | 
| --- | --- | --- |  --- | --- |  --- | --- |  --- | --- |  --- |--- |
|   MSR |     |     |      |     ||     |     |      |     |
|   Margin |     |     |      |     ||     |     |      |     |
|   Entropy  |     |     |      |     ||     |     |      |     |
|   MC-Dropout |     |     |      |     ||     |     |      |     |
|   Image Transformations |     |     |      |     ||     |     |      |     |
|   One-Class SVM |     |     |      |     ||     |     |      |     |
|   GMM |     |     |      |     ||     |     |      |     |
|   Density Forest|     |     |      |     ||     |     |      |     |

#### Zurich Dataset: Novelty Detection PR AUC
 
|   Method  |   Roads  |  Buildings   |   Trees  | Grass   | 
| --- | --- | --- |  --- | --- | 
|   MSR |   67.2  |     |      |     |
|   Margin |   60.2  |     |      |     |
|   Entropy  | 64.2 |     |      |     |
|   MC-Dropout | 60.5 |     |      |     |
|   Image Transformations |   |     |      |     |
|   One-Class SVM |  **75.0**   |     |      |     |
|   GMM |   33.8  |     |      |     |
|   Density Forest|   55.4  |     |      |     |

