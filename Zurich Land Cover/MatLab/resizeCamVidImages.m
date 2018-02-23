function imds = resizeCamVidImages(imds, imageFolder)
% Resize images to [360 480].

if ~exist(imageFolder,'dir')
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);

    % Resize image.
    I = imresize(I,[360 480]);

    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end