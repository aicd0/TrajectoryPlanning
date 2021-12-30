function robot = getRobot
    joint_limit = [-80 80]*pi/180;

    % DH params
    desc = [
        robotBodyDesc(0  , pi/2, 0, 0 , "revolute", [-pi pi], false)
        robotBodyDesc(0.5, 0   , 0, 0 , "revolute", joint_limit, true)
        robotBodyDesc(0.5, 0   , 0, 0 , "revolute", joint_limit, true)
        robotBodyDesc(0.5, 0   , 0, 0 , "revolute", joint_limit, true)
    ];
    
    % Assemble the robot.
    robot = rigidBodyTree("DataFormat", "column");
    num_bodies = size(desc, 1);
    bodies = cell(num_bodies, 1);
    joints = cell(num_bodies, 1);
    last_body = "base";
    
    for i = 1 : num_bodies
        % Setup the joint.
        joints{i} = rigidBodyJoint(['jnt' num2str(i)], desc(i).Type);
        joints{i}.PositionLimits = desc(i).Limits;
        setFixedTransform(joints{i}, desc(i).DH, "dh");

        % Setup the body.
        bodies{i} = rigidBody(['body' num2str(i)]);
        if i == num_bodies
            bodies{i}.Name = 'end_effector';
        end
        bodies{i}.Joint = joints{i};
    
        % Add collisions.
        if desc(i).Collision
            length = desc(i).A;
            tform = axang2tform([0 1 0 pi/2]) * trvec2tform([0 0 -length/2]);
            addCollision(bodies{i}, "cylinder", [0.05 length - 0.05], tform);
        end

        % Attach to the robot.
        addBody(robot, bodies{i}, last_body);
        last_body = bodies{i}.Name;
    end
end