% Gets the distances (i.e. the measured difference) between 
% the training images and the testing images.
%   INPUT:  d0: desired dimension for Eigenfaces. 400 x [32, 64]
%           d1: desired dimension for Fisherfaces. d0 x [8, 16, 24]
%           d2: desired dimension for Laplacianfaces. d0 x [8, 16, 24]
%           faces: 6690 x 400. Each row contains one image.
%           ids: 6690 x 1. Each row contains the corresponding id for an image.
%           train_images: 6267 x 400. 
%           test_images: 418 x 400. Contains one image for each unique id.
%           show: 1 to plot model faces, etc. 
%   OUTPUT: f_dist: 6267 x 418. The distances (i.e. the measured difference) 
%                   between the training images and the testing images.

function [f_dist] = getDistances(d0, d1, d2, faces, ids, train_images, test_images, show)
[A0, ~] = getEigenfacemodel(faces, d0, show);
if d1 ~= 0 % Fisherfaces
    A1 = getFisherfacemodel(faces, A0, ids, d1, show);
    x_train = train_images*A0*A1;
    x_test = test_images*A0*A1;
elseif d2 ~= 0 % Laplacianfaces
    A2 = getLaplcaianfacemodel(faces, A0, ids, d2, show);
    x_train = train_images*A0*A2;
    x_test = test_images*A0*A2; 
else % just Eigenfaces
    x_train = train_images*A0;
    x_test = test_images*A0; 
end

f_dist = pdist2(x_train, x_test);

end