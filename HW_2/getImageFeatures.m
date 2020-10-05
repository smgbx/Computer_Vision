%  im - input images, let us make them all grayscale only, so it is a  h x w matrix
%  opt.type = { ‘sift’, ‘dsft’} for sift and densesift
%  f - n x d matrix containing n features of d dimension
% DONE

function d = getImageFeatures(im, opt)
% run vl_feat setup
dbg=1;
if dbg
run('C:\Program Files\vlfeat-0.9.21\toolbox\vl_setup')
end

im = single(rgb2gray(imread(im)));
if opt == "sift"
    [~, d] = vl_sift(im);
elseif opt == "dsft" 
    [~, d] = vl_dsift(im);
end
return;

    
    