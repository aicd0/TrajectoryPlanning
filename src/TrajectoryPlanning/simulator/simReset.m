% Reset robot.
tmp.max_try = 100;
for i = 1 : tmp.max_try
    state.config = randomConfiguration(robot);
    tmp.colliding = any(checkCollision(robot, state.config, obstacles));
    if ~tmp.colliding
        break
    end
    assert(i < tmp.max_try);
end

state.achieved = getPos(robot, state.config);

% New stage.
simStage;