% features - n x d matrix containing training features from say 100 images. 
% kd - desired dimension of the feature (feature reduction)
% k - VLAD kmeans model number of cluster 
% vlad_km - VLAD kmeans model
% A - PCA projection for dimension reduction

function [vlad_km, A]=getVladModel(features, kd, k) % 16, 64
% Dimension reduction with PCA
A = pca(double(features));
reduced_features = (double(features))*A(:,1:kd); % 128 x kd

% VLAD requires "data-to-cluster assignments" to be passed in.
% Obtain visual word dictionary; "codebook" / "centroids"
vlad_km = vl_kmeans(reduced_features', k);
end
