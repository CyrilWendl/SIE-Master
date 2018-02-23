% cd to your unzipped folder containing this file
% e.g. cd('/path/to/Zurich_Summer/')
%% Retrive max - min - mean - av std information

warning('DID YOU READ THE README????')

MAXI  = -Inf(1,4);
MINI  = Inf(1,4);
MEANS = 0;
STDEV = 0; 

for i = 1:20
   I = single(imread(sprintf('images_tif/zh%i.tif',i))); 
   maxi = max(reshape(I,size(I,1)*size(I,2),[]),[],1);
   mini = min(reshape(I,size(I,1)*size^(I,2),[]),[],1);
   if sum(maxi > MAXI) > 0
       MAXI(maxi > MAXI) = maxi(maxi > MAXI);
   end
   if sum(mini < MINI) > 0
       MINI(mini < MINI) = mini(mini < MINI);
   end
   MEANS = MEANS + mean(reshape(I,size(I,1)*size(I,2) ,[]),1);
   STDEV = STDEV + std(reshape(I,size(I,1)*size(I,2),[]),[],1);
end

MEANS = MEANS./20;
STDEV = STDEV./20;

%% Visualize images and save raw and contrast enhanced .mat files locally

IMLOCALDIR = './images_matlab/';
if ~isdir(IMLOCALDIR)
    mkdir(IMLOCALDIR)
end

for i = 1:20
    I = double(imread(sprintf('./images_tif/zh%i.tif',i)));
    [x,y,d] = size(I);
    I = reshape(I,x*y,d);
    
    rescaledim = (I - repmat(MINI,size(I,1),1)) ./ repmat(MAXI-MINI,size(I,1),1);
    
    IM = reshape(uint16(rescaledim.*2^16),x,y,d);
    save(sprintf('%s zh%i_rawDN.mat',IMLOCALDIR,i),'IM')
    
%     figure(1); imshow(IM(:,:,[4 3 2]),[]); title('Raw DN, 16 bit')

    IMe = zeros(size(IM),'uint16');
    IMe(:,:,1:3) = imadjust(IM(:,:,[1 2 3]),stretchlim(IM(:,:,[1 2 3])));
    IMe(:,:,4)   = imadjust(IM(:,:,[4]),stretchlim(IM(:,:,[4])));

    save(sprintf('%s zh%i_CE.mat',IMLOCALDIR,i),'IMe')
    
%     figure(2); imshow(IMe(:,:,[4 3 2])); title('contrast enhanced, 16bit')
%     pause
end

%% Convert RGB ground truths to CLASS \in {1,...,C} raster format
LEGEND = [
    255    255    255;  % Background
      0      0      0;  % Roads
    100    100    100;  % Buildings
      0    125      0;  % Trees
      0    255      0;  % Grass
    150     80      0;  % Bare Soil
      0      0    150;  % Water
    255    255      0;  % Railways
    150    150    255]; % Swimming Pools 

GTLOCALDIR = './groundtruth_indexes/';
if ~isdir(GTLOCALDIR)
    mkdir(GTLOCALDIR)
end

for i = 1:20

    GT = imread(sprintf('groundtruth/zh%i_GT.tif',i));
    GT = uint8(rgb2label(GT,LEGEND));
    
    imwrite(GT,sprintf('%sGTZH%i_indMap.tif',GTLOCALDIR,i))

%     imagesc(GT); axis image; axis off; pause
end

%% Reconvert maps to uint8 plus geographical reference (i.e. write a geotiff 
%  with same geo-object as input image tiff (requires the mapping toolbox)

GTLOCALDIR = './maps_converted/';
if ~isdir(GTLOCALDIR)
    mkdir(GTLOCALDIR)
end

for i = 1:20
    
    [A,R] = geotiffread(['images_tif/zh' num2str(i) '.tif']);
    info  = geotiffinfo(['images_tif/zh' num2str(i) '.tif']);
    GT = imread(sprintf('groundtruth_indexes/GTZH%i_indMap.tif',i));
    GT = uint8(rgb2label(GT+1,LEGEND));
    geotiffwrite(sprintf('%szh%i_GT.tif',GTLOCALDIR,i),GT,R, ...
        'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
%     imshow(GT); pause
end
