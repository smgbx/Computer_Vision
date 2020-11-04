% A function that adds the extra images and their id (999)
% to the dataset and saves it to a new file.

function addImages
images = load('faces-ids-n6680-m417-20x20.mat');
old_faces = images.faces;
old_ids = images.ids;

new_images = load('ORL_32x32.mat');
new_faces = new_images.fea(1:10, :);

new_ids = zeros(10, 1)+999;
new_faces_scaled = rescale(new_faces, 0, 1);

new_faces = zeros(10, 400);
for k=1:10
    imwrite(reshape(new_faces_scaled(k, :), [32, 32]), 'original_image.png');
    i = imread('original_image.png');
    j = imresize(i, [20,20]);
    j = reshape(j, [1,400]);
    j = rescale(j, 0, 1);
    new_faces(k, :) = j;
end

faces = [old_faces;new_faces];
ids = [old_ids; new_ids];

save('faces_updated.mat', 'faces', 'ids');
end