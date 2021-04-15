function feature_map = extract_features(image, pos, scales, features, feature_info)

% Sample image patches at given position and scales. Then extract features
% from these patches.
% Requires that cell size and image sample size is set for each feature.

if ~iscell(features)
    error('Wrong input');
end

num_features = length(features);
num_scales = length(scales);

% img_sample_sz = extract_info.img_sample_sizes{sz_ind};
img_sample_raw = feature_info.img_sample_raw;
img_sample_sz = feature_info.img_sample_sz;
img_samples = zeros(img_sample_sz(1), img_sample_sz(2), size(image,3), num_scales, 'uint8');
for scale_ind = 1:num_scales
    img_samples(:,:,:,scale_ind) = sample_patch(image, pos, img_sample_sz*scales(scale_ind), img_sample_sz, feature_info.use_mexResize);
%     img_samples(:,:,:,scale_ind) = sample_patch(image, pos, img_sample_raw*scales(scale_ind), img_sample_sz, feature_info.use_mexResize);
end

% Find the number of feature blocks and total dimensionality
num_feature_blocks = 0;
total_dim = 0;
for feat_ind = 1:num_features
    num_feature_blocks = num_feature_blocks + length(features{feat_ind}.fparams.nDim);
    total_dim = total_dim + sum(features{feat_ind}.fparams.nDim);
end

if feature_info.is_cell
    feature_map = cell(1, 1, num_feature_blocks);
else
    feature_map = zeros(feature_info.data_sz(1), feature_info.data_sz(2), total_dim, num_scales, 'single');
end

% Extract feature maps for each feature in the list
ind = 1;
for feat_ind = 1:num_features
    feat = features{feat_ind};
    
    feature_map_ = feat.getFeature(img_samples, feat.fparams);
    
    % Do feature normalization
    feature_map_ = feature_map_ .* sqrt(size(feature_map_,1) * size(feature_map_,2) * size(feature_map_,3) ./ ...
        (sum(reshape(feature_map_, [], 1, 1, size(feature_map_,4)).^2, 1) + eps));
    
    if feature_info.is_cell
        num_blocks = 1;
        feature_map{ind} = feature_map_;
    else
        num_blocks = feat.fparams.nDim;
        feature_map(:,:,ind:ind+num_blocks-1,:) = feature_map_;
    end
    
    ind = ind + num_blocks;
end

end