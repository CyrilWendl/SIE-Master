clc, close, clear all

%% settings
ml_dir = '/Users/cyrilwendl/Documents/EPFL/Machine Learning/Machine-Learning-2017/project2/training'
outputFolder = fullfile(ml_dir, 'images');
imageDir = fullfile(ml_dir,'images');
labelDir = fullfile(ml_dir,'groundtruth');
classNames = {'background', 'road'}
labelIDs = [1, 2]

%% load data
imds = imageDatastore(imageDir);
imds.ReadFcn = @import_images; % do processing

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
pxds.ReadFcn = @import_labels;

%% show label frequency
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classNames),frequency)
xticks(1:numel(classNames))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% class weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

%% split test / validation data
[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partition_data(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

% create segnet
imageSize = [400 400 3];
numClasses = numel(classNames);
encoderDepth = 8
lgraph = segnetLayers(imageSize,numClasses,encoderDepth);

% new weighted class layer
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');

%options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-5, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress');
  
% image agmentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

plot(lgraph)


%%
im_test = imds.readall{1};
imshow(im_test)i

%% 

fun = @(block_struct) ...
   block_struct.data;

I2 = blockproc(im_test,[32 32],fun);

%% start training
datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,'DataAugmentation',augmenter);

%% train segnet

[net, info] = trainNetwork(datasource,lgraph,options);



%% 2. CNN
%% Read the 10th image and corresponding pixel label image.
I = readimage(imds,10);
C = readimage(pxds,10);

%% Read the 10th pixel label image and display it on top of the image.
cmap = jet(numel(classNames));
B = labeloverlay(I,C,'Colormap',cmap);
figure
imshow(B)

% add colorbar
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap)

%% net
numFilters = 32;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([400 400 3])%imageInputLayer([32 32 3])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)
    ]


% training options
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'InitialLearnRate', 1e-9, ...
    'MaxEpochs', 3, ...
    'MiniBatchSize', 2, ...
    'Verbose',false,...
    'Plots','training-progress');
    %'ValidationData',{valImages,valLabels},...
    %'ValidationFrequency',30,...


%% train network
net = trainNetwork(datasource,layers,opts);

%% read and display a test image
testImage = imread(fullfile(imageDir,'satImage_002.png'));

%figure
%imshow(testImage)

% Segment the test image and display the results.
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
