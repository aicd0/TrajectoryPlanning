for i=1 : size(action, 1)
    action(i) = max(-1, min(1, action(i))); %#ok<SAGROW>
end

action = action * 0.1;
tmp.config = state.config + action;

% Update collision.
state.collision = checkCollision(robot, tmp.config, obstacles);

% Update config.
if ~state.collision
    state.config = tmp.config;
end

% Update achieved.
state.achieved = getPos(robot, state.config);
