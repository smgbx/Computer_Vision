% f - n x d matrix containing training features from say 100 images. 
% kd - desired dimension of the feature (feature reduction)
% k - VLAD kmeans model number of cluster 
% vlad_km - VLAD kmeans model
% A - PCA projection for dimension reduction

function [vlad_km, A]=getVladModel(features, kd, k) % 16, 64
% flatten features into a matrix 128 x n matrix, where n is number of
% descriptors
all_features = double([features{:}]);

% dimension reduction with PCA
A = pca(double(all_features));
reduced_features = (double(all_features))*A(:,1:kd); % 128 x kd

% VLAD requires "data-to-cluster assignments" to be passed in.
% 1. Obtain visual word dictionary; "codebook"
centroids = vl_kmeans(reduced_features', k);
% 2. kd-tree = fast vector quantization technique
kdtree = vl_kdtreebuild(centroids); 
% 3. nn = indexes of the nearest centroid to each vector in features
nn = vl_kdtreequery(kdtree, centroids, reduced_features'); 

% Create assignment matrix that assigns each descriptor to a cluster
assignments = zeros(k, numel(reduced_features'));
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

% Encode using VLAD. 
vlad_km = vl_vlad(reduced_features', centroids, assignments); % (128 x n)(128 x k)(k x n)
end
