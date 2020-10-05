function [fv_gmm, A] = getFisherVectorModel(features, kd, k)
% Flatten features into a matrix 128 x n matrix, where n is number of
% descriptors
all_features = double([features{:}]);

% Dimension reduction with PCA
A = pca(all_features);
reduced_features = all_features*A(:,1:kd);

% Encode using GMM
[fv_gmm.m, fv_gmm.cov, fv_gmm.p] = vl_gmm(reduced_features', k);
end