% Gets the Laplacianface model. 
%   INPUT:  faces: 6690 x 400. Each row contains one image.
%           A0: Eigenface model
%           ids: 6690 x 1. Each row contains the corresponding id for an image.
%           d2: desired dimension for Laplacianfaces. d0 x [8, 16, 24]
%           show: 1 to plot model faces, etc. 
%   OUTPUT: A2: Laplacianface model
%           S: affinity matrix

function [A2, S2] = getLaplcaianfacemodel(faces, A0, ids, d2, show)
n_face = 1200;

% eigenface 
x1 = faces(1:n_face,:)*A0; 
ids = ids(1:n_face); 

% LPP - compute affinity
f_dist1 = pdist2(x1, x1);
% heat kernel size
mdist = mean(f_dist1(:)); 
h = -log(0.15)/mdist; 
S1 = exp(-h*f_dist1); 

% utilize supervised label info
id_dist = pdist2(ids, ids);

S2 = S1; 
% setting affinity to 0 for intra-class pairs
S2(id_dist~=0) = 0; 

% laplacian face
lpp_opt.PCARatio = .99; 
[A2, ~]=LPP(S2, lpp_opt, x1); 
A2 = A2(:,1:d2);

if show==1
    figure(32); subplot(2,2,1); imagesc(f_dist1); colormap('gray'); title('d(x_i, d_j)');
    subplot(2,2,2); imagesc(S1); colormap('gray'); title('affinity'); 
    subplot(2,2,3); imagesc(id_dist); title('label distance');
    subplot(2,2,4); imagesc(S2); colormap('gray'); title('affinity-supervised');  

    lapface = eye(400)*A0*A2; 
    for k=1:8 
       figure(37);
       subplot(2,4,k); imagesc(reshape(lapface(:,k),[20, 20])); colormap('gray');
       title(sprintf('lapf_%d', k)); 
    end
end

end