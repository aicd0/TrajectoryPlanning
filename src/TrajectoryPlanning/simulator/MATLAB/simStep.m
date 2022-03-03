tmp.config = state.config + action * 0.1;
for i = 1:length(robot.Bodies)
    joint_limits = robot.Bodies{1, i}.Joint.PositionLimits;
    tmp.config(i) = max(tmp.config(i), joint_limits(1));
    tmp.config(i) = min(tmp.config(i), joint_limits(2));
end

% Update state.collision
state.collision = checkCollision(robot, tmp.config, obstacles);

% Update state.config
if ~state.collision
    tmp.last_config = state.config;
    state.config = tmp.config;
end

% Update state.achieved
state.achieved = getPos(robot, state.config);

% Update state.deadlock
state.deadlock = sum(abs(state.config - tmp.last_config)) < 0.001 && sum(abs(action)) > 0.1;
