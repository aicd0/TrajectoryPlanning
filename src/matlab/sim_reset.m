% Initialize the robot.
st_config = randomConfiguration(robot);
st_achieved = getPos(robot, st_config);

% Start a new stage.
sim_stage;