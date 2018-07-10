# To Do

## Ideas
- Rewrite CNN in PyTorch for Dropout
- Use results of pre-softmax activations * w to predict same / different outcome on class
- Use parametric t-SNE for dimensionality reduction
- Apply DF to other dataset, e.g., CIFAR10

## PyTorch
- Test 64 vs 128 patches, 30 epochs (128: ~93%, ~71% vs ~87%, 73%)
- Avoid overfitting
  - Weight decay 
  - Lighter network
  - More dropout (before: 89%, 69%, after: 83%, 65%)
- Implement testing with dropout

## Folder structure
- Put all libraries in same subfolder

## Report
- Compare single tree to Density Forest