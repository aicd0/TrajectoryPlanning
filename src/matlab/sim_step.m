action = [1; -2; 3; -4];

for i=1 : size(action, 1)
    action(i) = max(-1, min(1, action(i)));
end

action = action * 0.01;
config = robot_config + action;

% Update colliding.
is_colliding = checkCollision(robot, robot_config, obstacles);
is_colliding = any(is_colliding);

% Update robot config.
robot_config = config;

% Update robot pos.
robot_pos = getPos(robot, robot_config);

% Output states.
clearvars state;
state.config = robot_config;
state.achieved = robot_pos;
state.target = targ_pos;
state.obstacle = obstacle_pos;
state.collide = is_colliding;