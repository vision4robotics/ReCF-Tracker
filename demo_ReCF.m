
% This demo script runs the ReCF tracker with hand-crafted features on the
% included "Surfing06" video (DTB70 dataset).

% Add paths
setup_paths();

% Load video information
video_path = './sequences/Surfing06'; % Surfing06 
[seq, ground_truth] = load_video_info(video_path);

% Run ReCF
results = run_ReCF(seq);

close all;