clc, clear all

%% Train Semantic Segmentation Network
dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

imds = imageDatastore(imageDir);

classNames = ["triangle","background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%% net
numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([32 32 1])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ]
%% training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64);

%% data source for training data
trainingData = pixelLabelImageSource(imds,pxds);
%% train network
net = trainNetwork(trainingData,layers,opts);

%% read and display a test image
testImage = imread('triangleTest.jpg');

figure
imshow(testImage)

%% Segment the test image and display the results.
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure
imshow(B)

%% class weight
tbl = countEachLabel(trainingData)
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
%% new
layers(end) = pixelClassificationLayer('ClassNames',tbl.Name,'ClassWeights',classWeights);

%% train network again
net = trainNetwork(trainingData,layers,opts);
classWeights = 1./frequency


%% Try to segment the test image again.
C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
figure
imshow(B)

%% Evaluate and inspect the results of semantic segmentation
dataSetDir = fullfile(toolboxdir('vision'), 'visiondata', 'triangleImages');
testImagesDir = fullfile(dataSetDir, 'testImages'); % Define the location of the test images.
imds = imageDatastore(testImagesDir); % Create an imageDatastore object holding the test images.
testLabelsDir = fullfile(dataSetDir, 'testLabels'); % Define the location of the ground truth labels.

%Define the class names and their associated label IDs. The label IDs are the pixel values used in the image files to represent each class.
classNames = ["triangle" "background"];
labelIDs   = [255 0];

% Create a pixelLabelDatastore object holding the ground truth pixel labels for the test images.
pxdsTruth = pixelLabelDatastore(testLabelsDir, classNames, labelIDs);
%% Load a semantic segmentation network that has been trained on the training images of triangleImages.
net = load('triangleSegmentationNetwork.mat');
net = net.net;

%%
pxdsResults = semanticseg(imds, net, "WriteLocation", tempdir);

%%
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruth);

%% 
metrics.ClassMetrics

%% 
metrics.ConfusionMatrix

normConfMatData = metrics.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames, classNames, 100 * normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title  = 'Normalized Confusion Matrix (%)';

%%
imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')

%%
evaluationMetrics = ["accuracy" "iou"];
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTruth, "Metrics", evaluationMetrics);
