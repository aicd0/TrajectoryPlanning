for i=1 : size(action, 1)
    action(i) = max(-1, min(1, action(i))); %#ok<SAGROW>
end

action = action * 0.09;
config = st_config + action;

% Update colliding.
st_collision = checkCollision(robot, config, obstacles);

% Update robot config.
if ~st_collision
    st_config = config;
end

% Update robot pos.
st_achieved = getPos(robot, st_config);

% Update state.
updateState;