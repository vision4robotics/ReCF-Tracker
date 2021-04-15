function seq = report_tracking_result(seq, result)

if strcmpi(seq.format, 'otb')
    seq.rect_position(seq.frame,:) = round([result.center_pos([2,1]) - (result.target_size([2,1]) - 1)/2, result.target_size([2,1])]);
else
    error('Uknown sequence format');
end