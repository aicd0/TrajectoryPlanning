gui = interactiveRigidBodyTree(robot, MarkerScaleFactor=0.5);

% Paint obstacles.
for i = 1:length(obstacles)
    show(obstacles{i});
end