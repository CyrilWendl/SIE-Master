%% function for getting categorical data from labels
function data = import_labels(im)
    data = imread(im)>125;
    data = data+1;
    %data = imresize(data,[36 36]);
    data = categorical(data,[1 2],{'background' 'road'});
end