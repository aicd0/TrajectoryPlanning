% Update desired.
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

% Update collision.
state.collision = checkCollision(robot, state.config, obstacles);