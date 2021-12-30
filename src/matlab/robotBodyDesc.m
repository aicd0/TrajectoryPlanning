classdef robotBodyDesc
    properties
        A
        Alpha
        D
        Theta
        Type
        DH
        Collide
        Limits
    end

    methods
        function obj = robotBodyDesc(a, alpha, d, theta, type, limits, collide)
            obj.A = a;
            obj.Alpha = alpha;
            obj.D = d;
            obj.Theta = theta;
            obj.Type = type;
            obj.Collide = collide;
            obj.Limits = limits;
        end

        function value = get.DH(obj)
        %get.DH
            value = [obj.A obj.Alpha obj.D obj.Theta];
        end
    end
end