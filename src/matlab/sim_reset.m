% Generate obstacles.
obstacle = collisionSphere(0.2);
obstacle_pos = randomPos(robot);
obstacle.Pose = trvec2tform(obstacle_pos);
obstacles = {obstacle};

% Generate a new goal.
while true
    config = randomConfiguration(robot);
    is_colliding = checkCollision(robot, config, obstacles);
    is_colliding = any(is_colliding);
    if ~is_colliding
        targ_pos = getPos(robot, config);
        break
    end
end

% Initialize the robot.
while true
    config = randomConfiguration(robot);
    is_colliding = checkCollision(robot, config, obstacles);
    is_colliding = any(is_colliding);
    if ~is_colliding
        robot_config = config;
        robot_pos = getPos(robot, config);
        break
    end
end

% Output states.
state = {robot_config; robot_pos; targ_pos; obstacle_pos; false};