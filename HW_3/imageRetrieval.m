% Retrieves the ids of the images that the models assume are associated
% with the image of the query id. 
%   INPUT:  d0: desired dimension for Eigenfaces. 400 x [32, 64]
%           d1: desired dimension for Fisherfaces. d0 x [8, 16, 24]
%           d2: desired dimension for Laplacianfaces. d0 x [8, 16, 24]
%           threshold: cutoff point for similarity. Images must have a 
%                      difference less than the threshold to be returned. 
%           show: 1 to plot model faces, etc. 
%   OUTPUT: ret_ids: n x 1. The returned ids of the query. 

function [ret_ids] = imageRetrieval(query_id, d0, d1, d2, threshold, show)
%clear all, close all, clc

% loading faces, ids, of n x 400 and n x 1
load faces_updated.mat;

n = length(ids); % total images
uid = unique(ids); 
m = length(uid); % unique persons

% query data index
q_indx = zeros(1, m);
for k = 1:m
    offs = find(ids == uid(k));
    q_indx(k) = offs(1);
end

% training data index
train_indx = setdiff((1:n), q_indx); % returns values in 1 thru n that is not in q_indx

t_num = length(train_indx);
train_images = zeros(t_num,400);
for i=1:t_num
    train_images(i,:) = (faces(train_indx(i), :));
end

q_num = length(q_indx);
test_images = zeros(q_num,400);
for i=1:q_num
    test_images(i,:) = faces(q_indx(i),:);
end

% NOTE: set EITHER d1 or d2 to one of the available dimensions. Set the
% other dimension to 0 to indicate that that model won't be used.
% If just using eigenfaces, set both d1 and d2 to 0.
f_dist = getDistances(d0, d1, d2, faces, ids, train_images, test_images, show);

% thresholded distances between every training and test image
f_dists_below_threshold = zeros(t_num, q_num)+100;
for i=1:q_num
    for j=1:t_num
         if f_dist(j,i) < threshold
            f_dists_below_threshold(j,i) = f_dist(j,i);
        end
    end
end

% find location of query image in query index
temp = find(ids==query_id);
query_indx = find(q_indx==temp(1));

% organize returned ids by closest
% key: train_image_index. value: dist from train image to test image
myMap = containers.Map(1:t_num, f_dists_below_threshold(1:t_num, query_indx));

% remove images with dist greater than threshold
for i=1:t_num
    if myMap(i) > threshold
        remove(myMap,i);
    end
end
indices = cell2mat(myMap.keys);
dists = cell2mat(myMap.values);

% sort dist in ascending order for mAP
[~, sortIdx] = sort(dists, "ascend");
sortedIndices = indices( sortIdx );

% get ids of matching images from training_indx
ret_ids = [];
for i=1:length(sortedIndices)
    ret_ids(end+1) = ids(train_indx(sortedIndices(i)));
end

end