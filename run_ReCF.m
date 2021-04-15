function results = run_ReCF(seq, res_path, bSaveImage)

% Initialize
params.seq = seq;

% Feature specific parameters
params.feature_info.cell_size = 4;
params.feature_info.use_mexResize = true;

cn_params.useForGray = false;
cn_params.tablename = 'CNnorm';
% Which features to include
params.t_features = {
    struct('getFeature',@get_fhog, 'fparams', []),...
    struct('getFeature',@get_gray, 'fparams', []),...
    struct('getFeature',@get_table_feature, 'fparams', cn_params),...
};

% Image sample parameters
params.search_area_scale = 4;           % The scaling of the target size to get the search area
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Detection parameters
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image
params.output_sigma_factor = 1/16;		% Label function sigma

% Regularization window
params.reg_window_max = 1e5;  % The maximum value of the regularization window
params.reg_window_min = 1e-3; % the minimum value of the regularization window

% Scale parameters for the translation model
% Only used if: params.use_scale_filter = false
params.number_of_scales = 7;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor

% Scale filter parameters
% Only used if: params.use_scale_filter = true
params.use_scale_filter = true;         % Use the fDSST scale filter or not (for speed)
params.scale_sigma_factor = 1/16;       % Scale label function sigma
params.scale_learning_rate = 0.024;		% Scale filter learning rate
params.number_of_scales_filter = 11;    % Number of scales
params.number_of_interp_scales = 33;    % Number of interpolated scales
params.scale_step_filter = 1.03;        % The scale factor for the scale filter
params.scale_model_max_area = 32*16;    % Maximume area for the scale sample patch
params.lambda = 1e-2;					% Scale filter regularization
params.do_feat_compress = true;
params.do_poly_interp = true;           % Do 2nd order polynomial interpolation to obtain more accurate scale

% ReCF parameters
params.learning_rate = 1;       % learning rate
params.gamma_I = 102.2;          % Parameter on Inferred response regularization
params.gamma_H = 28;          % Parameter on historical response regularizatio

% ADMM parameters
params.admm_iterations = 3;   % Iterations
params.mu = 100;              % Initial penalty factor
params.beta = 500;            % Scale step
params.mu_max = 100000;       % Maximum penalty factor

% Visualization
params.print_screen = 1;     % Print time spent on each frame to the command window
params.disp_fps = 1;         % Display fps info when tracking process ends
params.visualization = 1;    % Visualize tracking and detection scores

% Run tracker
results = tracker(params);
