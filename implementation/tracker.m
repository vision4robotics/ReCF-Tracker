function results = tracker(params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return
end

% Define data types
params.data_type = zeros(1, 'single');
feature_info = params.feature_info;
feature_info.data_type = params.data_type;

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end

% Calculate search area and initial scale factor
search_area = prod(target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% square search area
img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); 

[features, feature_info] = init_features(params.t_features, feature_info, is_color_image, img_sample_sz);

% Set feature info
img_support_sz = feature_info.img_support_sz;
cell_sz = feature_info.cell_size;

% The size of the label function DFT. Equal to the maximum filter size.
output_sz = feature_info.data_sz;

% Construct the Gaussian label function
output_sigma = sqrt(prod(floor(base_target_sz/cell_sz))) * params.output_sigma_factor;
rg           = circshift(-floor((output_sz(1)-1)/2):ceil((output_sz(1)-1)/2), [0 -floor((output_sz(1)-1)/2)]);
cg           = circshift(-floor((output_sz(2)-1)/2):ceil((output_sz(2)-1)/2), [0 -floor((output_sz(2)-1)/2)]);
[rs, cs]     = ndgrid(rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = fft2(y);
yf = cast(yf, 'like', params.data_type);

% construct cosine window
cos_window = hann(output_sz(1)+2) * hann(output_sz(2)+2)';
cos_window = cast(cos_window(2:end-1,2:end-1), 'like', params.data_type);

% Define spatial regularization windows
reg_window = construct_regwindow(params, output_sz, base_target_sz/cell_sz);

% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((output_sz(1) - 1)/2) : ceil((output_sz(1) - 1)/2), [0, -floor((output_sz(1) - 1)/2)]);
kx = circshift(-floor((output_sz(2) - 1)/2) : ceil((output_sz(2) - 1)/2), [0, -floor((output_sz(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

if params.use_scale_filter
    [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
else
    % Use the translation filter to estimate the scale.
    nScales = params.number_of_scales;
    scale_step = params.scale_step;
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
end

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    frame_tic = tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        
        %translation search
        % Extract features at multiple resolutions
        sample_pos = round(pos);
        sample_scale = currentScaleFactor*scaleFactors;
        xt = extract_features(im, sample_pos, sample_scale, features, feature_info);
        
        % Do windowing of features
        xtw = xt .* cos_window;
        xtf = fft2(xtw);
        
        response_f_sum = sum(conj(wf) .* xtf, 3);
        
        % Also sum over all feature blocks.
        % Gives the fourier coefficients of the convolution response.
        response_f = permute(gather(response_f_sum), [1 2 4 3]);
        response = ifft2(response_f, 'symmetric');
        [trans_row, trans_col, scale_ind] = resp_newton(response, response_f, newton_iterations, ky, kx, output_sz);
        
        % Compute the translation vector in pixel-coordinates and round
        % to the closest integer pixel.
        translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) * currentScaleFactor * scaleFactors(scale_ind);
        scale_change_factor = scaleFactors(scale_ind);
        
        % update position
        pos = sample_pos + translation_vec;
        
        % Advoid error
        if params.clamp_position
            pos = max([1 1], min([size(im,1) size(im,2)], pos));
        else
            if ~all(~isnan(pos))
                pos = [size(im,1)/2, size(im,2)/2];
            end
        end
        
        % Do scale tracking with the scale filter
        if nScales > 0 && params.use_scale_filter
            scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
        end
        
        % Update the scale
        currentScaleFactor = currentScaleFactor * scale_change_factor;
        
        % Adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Extract sample
    % Extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features(im, sample_pos, currentScaleFactor, features, feature_info);
    
    % Do windowing of features
    xlw = xl .* cos_window;
    xlf = fft2(xlw);

    if (seq.frame == 1)
        xf_p = zeros(size(xlf));
        wf_p = zeros(size(xlf));
        model_xf = xlf;
    else
		xf_p = model_xf;
        wf_p = wf;
        model_xf = (1 - params.learning_rate) * model_xf + params.learning_rate * xlf;
    end
    
    % Do training
    wf = train_ReCF(params, model_xf, yf, reg_window, xf_p, wf_p);
    
    % Update the scale filter
    if nScales > 0 && params.use_scale_filter
        scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
    end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz * currentScaleFactor;
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
%     fprintf('[%d: %.5f, %.5f, %.5f, %.5f]\n', seq.frame, tracking_result.center_pos, tracking_result.target_size)
    
    curr_t = toc(frame_tic);
    seq.time = seq.time + curr_t;
    if params.print_screen == 1
        if seq.frame == 1
            fprintf('initialize: %f sec.\n', curr_t);
            fprintf('===================\n');
        else
            fprintf('[%04d/%04d] time: %f\n', seq.frame, seq.num_frames, curr_t);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else
            figure(fig_handle);
            imagesc(im_to_show);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
        end
        drawnow
    end
end
% close(writer);
[seq, results] = get_sequence_results(seq);

if params.disp_fps
    disp(['fps: ' num2str(results.fps)]);
end

