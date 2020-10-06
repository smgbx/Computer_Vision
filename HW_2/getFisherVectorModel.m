% features - n x d matrix containing training features from say 100 images. 
% kd - desired dimension of the feature (feature reduction)
% k - number of cluster 
% vlad_km - VLAD kmeans model
% A - PCA projection for dimension reduction

function [fv_gmm, A] = getFisherVectorModel(features, kd, k)
% Dimension reduction with PCA
A = pca(double(features));
reduced_features = double(features)*A(:,1:kd);

% Encode using GMM
[fv_gmm.m, fv_gmm.cov, fv_gmm.p] = vl_gmm(reduced_features', k);
end