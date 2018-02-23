clc, close, clear all
%% Setup
vgg16();
% In addition, download a pretrained version of SegNet.

pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/segnetVGG16CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedSegNet');
pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat');
if ~exist(pretrainedFolder,'dir')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained SegNet (107 MB)...');
    websave(pretrainedSegNet,pretrainedURL);
end


%%  Download data
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

outputFolder = fullfile(tempdir, 'CamVid');
imageDir = fullfile(outputFolder,'images');
labelDir = fullfile(outputFolder,'labels');

if ~exist(outputFolder, 'dir')
    disp('Downloading 557 MB CamVid data set...');
    
    unzip(imageURL, imageDir);
    unzip(labelURL, labelDir);
end
%% Use imageDatastore to load CamVid images
imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);

%% Display one of the images.

I = readimage(imds, 1);
I = histeq(I);
figure
imshow(I)

%% Load CamVid Pixel-Labeled Images
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

% reduce 32 classes into 11
labelIDs = camvidPixelLabelIDs();

% Use the classes and label IDs to create the pixelLabelDatastore:
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

% Read and display one of the pixel-labeled images by overlaying it on top of an image.

C = readimage(pxds, 1);

cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

figure
imshow(B)
pixelLabelColorbar(cmap,classes);
%% Analyze Dataset Statistics
% To see the distribution of class labels in the CamVid dataset, use countEachLabel. This function counts the number of pixels by class label.

tbl = countEachLabel(pxds)
pixelLabelColorbar(cmap,classes);

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')


%% Prepare Training and Test Sets
% Resize CamVid Data
imageFolder = fullfile(outputFolder,'imagesReszed',filesep);
imds = resizeCamVidImages(imds,imageFolder);

labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizeCamVidPixelLabels(pxds,labelFolder);

% Prepare Training and Test Sets
[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionCamVidData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%% Create the Network
imageSize = [360 480 3];
numClasses = numel(classes);
lgraph = segnetLayers(imageSize,numClasses,'vgg16');

% balance classes
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');

options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2);

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

plot(lgraph)
%% Train the model
% not enough memory available for training on 2 GB GPU
doTraining = false;
if doTraining
    [net, info] = trainNetwork(datasource,lgraph,options);
else
    data = load(pretrainedSegNet);
    net = data.net;
end

%% Test Network on One Image

I = read(imdsTest);
C = semanticseg(I, net);

% Display the results.

B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);

% Compare the results in C with the expected ground truth stored in pxdsTest. 
expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);


imshowpair(actual, expected)
%% Accuracy measures
iou = jaccard(C, expectedResult);
table(classes,iou)

%% Accuracy on entire image
%semanticseg returns the results for the test set as a pixelLabelDatastore object. The actual pixel label data for each test image in imdsTest is written to disk in the location specified by the 'WriteLocation' parameter. Use evaluateSemanticSegmentation to measure semantic segmentation metrics on the test set results.
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);

% evaluateSemanticSegmentation returns various metrics for the entire dataset, for individual classes, and for each test image. To see the dataset level metrics, inspect metrics.DataSetMetrics .
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
%% Show metrics

metrics.DataSetMetrics
metrics.ClassMetrics
