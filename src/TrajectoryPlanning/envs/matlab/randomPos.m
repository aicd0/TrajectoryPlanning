function pos = randomPos(robot)
    config = randomConfiguration(robot);
    pos = getPos(robot, config);
end
