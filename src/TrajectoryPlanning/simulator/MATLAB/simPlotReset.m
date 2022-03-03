% Paint obstacles.
for i = 1:length(obstacles)
    show(obstacles{i});
end

% Paint target point.
plot3(state.desired(1), state.desired(2), state.desired(3), 'b.');

% Paint robot.
simPlotStep;
