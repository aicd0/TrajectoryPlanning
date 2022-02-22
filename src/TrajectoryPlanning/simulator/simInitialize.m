clearvars;

% Initialize robot.
robot = getRobot();

% Initialize obstacles.
tmp.obstacle = collisionSphere(0.2);
obstacles = {tmp.obstacle};

tmp.max_try = 100;
for i = 1 : tmp.max_try
    tmp.obstacle.Pose = trvec2tform(randomPos(robot));
    tmp.config = randomConfiguration(robot);
    tmp.colliding = any(checkCollision(robot, tmp.config, obstacles));
    if ~tmp.colliding
        break
    end
    assert(i < tmp.max_try);
end