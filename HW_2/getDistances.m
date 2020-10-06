% pairs - 1 x m matrix of corresponding pairs
% feat - n x d matrix containing training features from say 100 images. 
% kd - desired dimension of the feature (feature reduction)
% k - VLAD kmeans model number of cluster 
% model - trained model; either vlad or fv

function[distances] = getDistances(pairs, feat, agg, kd, k, model)
distances=zeros(1,25);
for i = 1:size(pairs,1)
    %get file names
    im1 = ['cdvs_thumbnails/cdvs-',num2str(pairs(i,1)),'.jpg'];
    im2 = ['cdvs_thumbnails/cdvs-',num2str(pairs(i,2)),'.jpg'];

    feat1 = getImageFeatures(im1, feat);
    feat2 = getImageFeatures(im2, feat);
    
    if agg == "vlad"
        agg1 = getVladAggregation(model, feat1, kd, k);
        agg2 = getVladAggregation(model, feat2, kd, k);
    elseif agg == "fv"
        agg1 = getFisherVectorAggregation(model, feat1, kd);
        agg2 = getFisherVectorAggregation(model, feat2, kd);
    end
   
    dist = pdist2(agg1, agg2);
    mean = mean2(dist);
    distances(1,i) = mean;
end