% feature : n x d matrix containing a feature from an image; output of
% getImageFeature(im,...)
% vlad_km : VLAD kmeans model

function [vlad] = getVladAggregation(vlad_km, features, kd, k)
% Dimension reduction with PCA
A = pca(double(features));
reduced_features = (double(features))*A(:,1:kd); % 128 x kd
% kd-tree = fast vector quantization technique of centroids
kdtree = vl_kdtreebuild(vlad_km); 
% nn = indexes of the nearest centroid to each vector in features
nn = vl_kdtreequery(kdtree, vlad_km, reduced_features'); 
% Create assignment matrix that assigns each descriptor to a cluster
assignments = zeros(k, numel(reduced_features'));
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;

% Encode using VLAD.
vlad = vl_vlad(reduced_features', vlad_km, assignments); % (128 x n)(128 x k)(k x n)
end