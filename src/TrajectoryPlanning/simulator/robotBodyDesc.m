classdef robotBodyDesc
    properties
        A
        Alpha
        D
        Theta
        Type
        DH
        Collision
        Limits
    end

    methods
        function obj = robotBodyDesc(a, alpha, d, theta, type, limits, collision)
            obj.A = a;
            obj.Alpha = alpha;
            obj.D = d;
            obj.Theta = theta;
            obj.Type = type;
            obj.Collision = collision;
            obj.Limits = limits;
        end

        function value = get.DH(obj)
        %get.DH
            value = [obj.A obj.Alpha obj.D obj.Theta];
        end
    end
end
