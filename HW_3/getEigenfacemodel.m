% Gets the Eigenface model. 
%   INPUT:  faces: 6690 x 400. Each row contains one image.
%           d0: desired dimension for Eigenfaces. 400 x [32, 64]
%           show: 1 to plot model faces, etc. 
%   OUTPUT: A0: Eigenface model
%           eigv: Eigenfaces

function [A0, eigv] = getEigenfacemodel(faces, d0, show)
% PCA:
[A0, ~, eigv] = pca(faces); % U, S, V
A0 = A0(:, 1:d0);

if show==1
    % eigenface basis
    h=20; w=20;
    figure(31); 
    for k=1:8
        subplot(2,4,k); colormap('gray'); imagesc(reshape(A0(:,k), [h, w]));
        title(sprintf('eigf_%d', k));
    end

    % eigenface projection
    nface=1200;
    x = faces*A0(:, 1:d0);
    f_dist = pdist2(x(1:nface,:), x(1:nface,:)); % distance between each image and all other images
    figure(32);
    imagesc(f_dist); colormap('gray');

    % displaying numerical eigenvalues to
    % demonstrate information gain/loss
    figure(30);
    subplot(1,2,1); grid on; hold on; stem(eigv, '.');
    f_eng=eigv.*eigv;
    subplot(1,2,2); grid on;
    hold on; plot(cumsum(f_eng)/sum(f_eng), '.-');
end

end