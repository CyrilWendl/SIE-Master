% function [GTlab] = rgb2label(GT,legend)
% simple function to convert an RGB label image to indexes and vice versa.
% Indexes must be in the same order as they appear in the legend entries, 
% provide legend in RGB 0-255 format values (standard matlab colormap).
% _____________________________________________________
% Michele Volpi, CALVIN, University of Edinbrugh

function [GTlab] = rgb2label(GT,legend)

if size(GT,3) == 3
    GTlab = zeros(size(GT,1)*size(GT,2),1);
    GTc = reshape(GT,size(GT,1)*size(GT,2),size(GT,3));
    
    for i = 1:size(legend,1)
        GTlab(GTc(:,1) == legend(i,1) & GTc(:,2) == legend(i,2) & GTc(:,3) == legend(i,3)) = i-1;
    end
    GTlab = reshape(GTlab,size(GT,1),size(GT,2));
else
    GTlab = zeros(size(GT,1)*size(GT,2),3);
    GTc = GT(:);
    
    for i = 1:size(legend,1)
        GTlab(GTc == i,:) = repmat([legend(i,1) legend(i,2) legend(i,3)],sum(GTc == i),1);
    end
    GTlab = reshape(GTlab,size(GT,1),size(GT,2),3);
end
