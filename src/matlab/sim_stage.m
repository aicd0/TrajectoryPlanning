% Generate obstacles.
obstacle = collisionSphere(0.5);
obstacles = {obstacle};

max_try = 100;
for i = 1 : max_try
    st_obstacle = randomPos(robot);
    obstacle.Pose = trvec2tform(st_obstacle);
    is_colliding = checkCollision(robot, st_config, obstacles);
    is_colliding = any(is_colliding);
    if ~is_colliding
        break
    end
    assert(i < max_try);
end

% Generate a new goal.
max_try = 100;
for i = 1 : max_try
    config = randomConfiguration(robot);
    is_colliding = checkCollision(robot, config, obstacles);
    is_colliding = any(is_colliding);
    if ~is_colliding
        st_desired = getPos(robot, config);
        break
    end
    assert(i < max_try);
end

% Update colliding.
st_collision = checkCollision(robot, st_config, obstacles);

% Update state.
updateState;