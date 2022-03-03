clearvars;

% Initialize robot.
tmp.robot_file = '../../../outputs/robot.mat';

if exist(tmp.robot_file, 'file')
    load(tmp.robot_file, 'robot');
    disp('Robot loaded.');
else
    robot = getRobot();
    save(tmp.robot_file, 'robot');
    disp('Robot created.');
end

% Generate obstacles.
tmp.obstacle_file = '../../../outputs/obstacles.mat';

if exist(tmp.obstacle_file, 'file')
    load(tmp.obstacle_file, 'obstacles');
    disp('Environment loaded.');
else
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
    save(tmp.obstacle_file, 'obstacles');
    disp('Environment created.');
end
