% Reset state.config
tmp.max_try = 100;
for i = 1 : tmp.max_try
    state.config = randomConfiguration(robot);
    tmp.colliding = any(checkCollision(robot, state.config, obstacles));
    if ~tmp.colliding
        break
    end
    assert(i < tmp.max_try);
end

% Update state.achieved
state.achieved = getPos(robot, state.config);

% Update state.desired
tmp.max_try = 100;
for i = 1 : tmp.max_try
    tmp.config = randomConfiguration(robot);
    tmp.colliding = any(checkCollision(robot, tmp.config, obstacles));
    if ~tmp.colliding
        state.desired = getPos(robot, tmp.config);
        break
    end
    assert(i < tmp.max_try);
end

% Update state.collision & state.deadlock
state.collision = checkCollision(robot, state.config, obstacles);
state.deadlock = false;
