function [features, feature_info] = init_features(features, feature_info, is_color_image, img_sample_sz)

feature_info.is_cell = false;

% find which features to keep
feat_ind = false(length(features),1);
for n = 1:length(features)
    
    if ~isfield(features{n}.fparams,'useForColor')
        features{n}.fparams.useForColor = true;
    end
    
    if ~isfield(features{n}.fparams,'useForGray')
        features{n}.fparams.useForGray = true;
    end
    
    if (features{n}.fparams.useForColor && is_color_image) || (features{n}.fparams.useForGray && ~is_color_image)
        % keep feature
        feat_ind(n) = true;
    end
end

% remove features that are not used
features = features(feat_ind);
num_features = length(features);

% Initialize features by
% - setting the dimension (nDim)
% - specifying if a cell array is returned (is_cell)
% - setting default values of missing feature-specific parameters
% - loading and initializing necessary data (e.g. the lookup table or the network)
for k = 1:length(features)
    if isequal(features{k}.getFeature, @get_fhog)
        if ~isfield(features{k}.fparams, 'nOrients')
            features{k}.fparams.nOrients = 9;
        end
        features{k}.fparams.nDim = 3*features{k}.fparams.nOrients+5-1;
        
    elseif isequal(features{k}.getFeature, @get_table_feature)
        table = load(['lookup_tables/' features{k}.fparams.tablename]);
        features{k}.fparams.nDim = size(table.(features{k}.fparams.tablename),2);
        
    elseif isequal(features{k}.getFeature, @get_gray)
        features{k}.fparams.nDim = 1;
    else
        error('Unknown feature type');
    end
    
    % Set default cell size
    if ~isfield(features{k}.fparams, 'cell_size')
        features{k}.fparams.cell_size = feature_info.cell_size;
    end
end

% Set feature info
feature_info.dim_block = cell(num_features,1);
feature_info.penalty_block = cell(num_features,1);

for k = 1:length(features)
    % update feature info
    feature_info.dim_block{k} = features{k}.fparams.nDim;
end
% Feature info for each cell block
feature_info.dim = cell2mat(feature_info.dim_block);

feature_info.img_sample_raw = round(img_sample_sz);

% Check the size with the largest number of odd dimensions (choices in the
% third dimension)
new_img_sample_sz = (1 + 2*round(img_sample_sz / (2*feature_info.cell_size))) * feature_info.cell_size;
feature_sz_choices = floor(bsxfun(@rdivide, bsxfun(@plus, new_img_sample_sz, reshape(0:feature_info.cell_size-1, 1, 1, [])), feature_info.cell_size));
num_odd_dimensions = sum(sum(mod(feature_sz_choices, 2) == 1, 1), 2);
[~, best_choice] = max(num_odd_dimensions(:));
pixels_added = best_choice - 1;
feature_info.img_sample_sz = round(new_img_sample_sz + pixels_added);
feature_info.img_support_sz = feature_info.img_sample_sz;

% Set the sample size and data size for each feature
feature_info.data_sz_block = cell(num_features,1);
for k = 1:length(features)
    features{k}.img_sample_sz = feature_info.img_sample_sz;
end

feature_info.data_sz = feature_info.img_sample_sz / feature_info.cell_size;