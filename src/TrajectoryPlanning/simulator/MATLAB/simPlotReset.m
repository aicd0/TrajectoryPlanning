% Paint target point.
if exist('last_targ_plt', 'var')
    delete(last_targ_plt);
end
last_targ_plt = plot3(state.desired(1), state.desired(2), state.desired(3), 'b.');

% Paint robot.
simPlotStep;