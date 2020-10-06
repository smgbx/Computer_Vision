% feature : n x d matrix containing a feature from an image; output of
% getImageFeature(im,...)
% vlad_km : VLAD kmeans model

function [fv] = getFisherVectorAggregation(fv_gmm, feature, kd)
% Dimension reduction with PCA
A = pca(double(feature));
reduced_feature = (double(feature))*A(:,1:kd); % 128 x kd

fv = vl_fisher(reduced_feature', fv_gmm.m, fv_gmm.cov, fv_gmm.p);
end