% Gets the Fisherface model. 
%   INPUT:  faces: 6690 x 400. Each row contains one image.
%           A0: Eigenface model
%           ids: 6690 x 1. Each row contains the corresponding id for an image.
%           d1: desired dimension for Fisherfaces. d0 x [8, 16, 24]
%           show: 1 to plot model faces, etc. 
%   OUTPUT: A1: Fisherface model

function [A1] = getFisherfacemodel(faces, A0, ids, d1, show)
n_face = 1200;
opt.Fisherface = 1;

% [A, eigv] = LDA(label, opt, data)
[A1, ~] = LDA(ids(1:n_face), opt, faces(1:n_face,:)*A0);
A1 = A1(:, 1:d1);

if show==1
    % fisherface basis
    h=20; w=20;
    fishface = eye(400)*A0*A1; 
    for k=1:8
       figure(37);
       subplot(2,4,k); imagesc(reshape(fishface(:,k),[h, w])); colormap('gray');
       title(sprintf('fisherf_%d', k)); 
    end

end