function pos = getPos(robot, config)
    tform = getTransform(robot, config, 'end_effector');
    pos = tform2trvec(tform);
end