gui = interactiveRigidBodyTree(robot, MarkerScaleFactor=0.5);
show(obstacle);
show(robot, state.config, 'PreservePlot', false, 'Collisions', 'on', 'Visuals', 'off');