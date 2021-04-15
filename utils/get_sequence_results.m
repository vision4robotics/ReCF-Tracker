function [seq, results] = get_sequence_results(seq)

if strcmpi(seq.format, 'otb')
    results.type = 'rect';
    results.res = seq.rect_position;
else
    error('Uknown sequence format');
end

if isfield(seq, 'time')
    results.fps = seq.num_frames / seq.time;
else
    results.fps = NaN;
end