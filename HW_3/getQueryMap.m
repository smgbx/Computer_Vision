% Compues the average precision from a single query
%   INPUT: q: query (actual) ids
%          retrv_ids: retrieved ids of images, sorted 
%          from most relevant (lowest dist) to least
%  OUTPUT: map: mAP score of query

function [map] = getQueryMap(q, retrv_ids)
retrieved_docs = length(retrv_ids);
tp = 0;
precision = 0;

for i=1:retrieved_docs
    if retrv_ids(i) == q
        tp = tp + 1;
        precision = precision + tp/i;
    end
end

ap = precision/tp;
map = ap/retrieved_docs;
end


