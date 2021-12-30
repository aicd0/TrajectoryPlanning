% main.m

robot = getRobot();
gui = interactiveRigidBodyTree(robot, MarkerScaleFactor=0.5);
obstacle = collisionSphere(0.2);
obstacle.Pose = trvec2tform([0.5 0.5 0.5]);
obstacles = {
    obstacle
};

show(obstacle);
txt = text(-0.5, 0.5, -0.5, "");

while true
    % Update robot configurations.
    config = gui.Configuration;

    % Check collision. 
    is_colliding = checkCollision(robot, config, obstacles);
    is_colliding = any(is_colliding);

    % Plot.
    show(robot, config, 'PreservePlot', false, 'Collisions', 'on', 'Visuals', 'off');
    txt.String = "isColliding: " + string(is_colliding);

    pause(1);
end