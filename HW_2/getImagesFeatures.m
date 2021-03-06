%  im - input images, let us make them all grayscale only, so it is a  h x w matrix
%  ASSUMING WE HAVE 100 RANDOM IMAGES ALREADY
%  opt.type = { ‘sift’, ‘dsft’} for sift and densesift
%  f - n x d matrix containing n features of d dimension
% DONE

function [descriptors] = getImagesFeatures(folder, opt)
% run vl_feat setup
dbg=1;
if dbg
run('C:\Program Files\vlfeat-0.9.21\toolbox\vl_setup')
end

% folder = 'test_images\'
filelist = dir(fullfile(folder,'*.jpg'));
nFiles = 100;
descriptors = {};

for i=1:nFiles
    im = imread(fullfile(folder, filelist(i).name));
    im = single(rgb2gray(im));
    if opt == "sift"
        [~, d] = vl_sift(im);
    elseif opt == "dsft" 
        [~, d] = vl_dsift(im);
    end
    descriptors{i} = d;
end
% Flatten features into a matrix 128 x n matrix, where n is number of
% descriptors
descriptors = double([descriptors{:}]);
return;

    
    