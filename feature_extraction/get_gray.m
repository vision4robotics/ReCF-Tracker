function [ feature_map ] = get_gray(im, fparam)

% Get a color space feature. Currently implements 'gray'.
[im_height, im_width, ~, num_images] = size(im);
feature_map = zeros(im_height, im_width, fparam.nDim, num_images, 'single');

single_im = single(im)/255;

for k = 1:num_images
%     feature_map(:,:,:,k) = single(rgb2gray(im(:,:,:,k)))/255 - 0.5;
    feature_map(:,:,:,k) =  rgb2gray(single_im(:,:,:,k)) - 0.5;
end

if fparam.cell_size > 1
    feature_map = average_feature_region(feature_map, fparam.cell_size);
end
end
