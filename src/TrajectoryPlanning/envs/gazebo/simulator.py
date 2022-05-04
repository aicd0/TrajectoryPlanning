import config
import framework.geometry as geo
import functools
import numpy as np
import sys
import threading
import time
import utils.platform
import utils.string_utils
from .reward import GazeboReward
from .state import GazeboState
from envs.simulator import Simulator
from framework.robot import Robot1
from framework.workspace import Workspace
from math import pi
from typing import Any, Type

# Import ROS and Gazebo packages.
if utils.platform.is_windows():
    sys.path.append(config.Environment.Gazebo.ROSLibPath)
    sys.path.append(config.Environment.Gazebo.ProjectLibPath)
import rospy
from gazebo_msgs.msg import ContactsState, LinkStates
from geometry_msgs.msg import Point
from robot_sim.srv import PlaceMarkers, StepWorld
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class SpinThread(threading.Thread):
    def run(self):
        if not rospy.core.is_initialized():
            raise rospy.exceptions.ROSInitException("client code must call rospy.init_node() first")
        self.__terminate_requested = False
        while not rospy.core.is_shutdown() and not self.__terminate_requested:
            rospy.rostime.wallsleep(0.5)
    
    def terminate(self) -> None:
        self.__terminate_requested = True
        while self.is_alive():
            time.sleep(0.5)

class ServiceLibrary:
    step_world = '/user/step_world'
    place_markers = '/user/place_markers'

class TopicLibrary:
    joint0_com = '/robot/joint0_position_controller/command'
    joint1_com = '/robot/joint1_position_controller/command'
    joint2_com = '/robot/joint2_position_controller/command'
    joint3_com = '/robot/joint3_position_controller/command'
    joint4_com = '/robot/joint4_position_controller/command'
    joint_states = '/robot/joint_states'
    link1_bumper = '/link1_bumper'
    link2_bumper = '/link2_bumper'
    link3_bumper = '/link3_bumper'
    link4_bumper = '/link4_bumper'
    link_states = '/gazebo/link_states'

class Gazebo(Simulator):
    __client_activated = False

    def __init__(self, name: str=None):
        super().__init__(name)

        if Gazebo.__client_activated:
            raise Exception()
        Gazebo.__client_activated = True

        self.robot = Robot1()
        self.workspace = Workspace()
        self.obstacles = [
            geo.Box(np.array([0, 0, 0.1]), np.array([1.1, 1.1, 0.3])),
            geo.Box(np.array([ 0.701,  0.701, 2]), np.array([0.4, 0.4, 4])),
            geo.Box(np.array([ 0.701, -0.701, 2]), np.array([0.4, 0.4, 4])),
            geo.Box(np.array([-0.701,  0.701, 2]), np.array([0.4, 0.4, 4])),
            geo.Box(np.array([-0.701, -0.701, 2]), np.array([0.4, 0.4, 4])),
        ]
        self.__desired = None

        # Load configs
        self.action_amp = self.configs.get(config.Environment.Gazebo.ActionAmp_)
        workspace_name = self.configs.get(config.Environment.Gazebo.Workspace_)
        workspace_min_r = self.configs.get(config.Environment.Gazebo.WorkspaceMinR_)

        # Load/Make workspace.
        if not self.workspace.load(workspace_name):
            joint_low = self.robot.joint_limits[0]
            joint_high = self.robot.joint_limits[1]
            joint_positions = [
                [p for p in np.arange(joint_low[0], joint_high[0], 6 * pi/180)],
                [p for p in np.arange(joint_low[1], joint_high[1], 6 * pi/180)],
                [p for p in np.arange(joint_low[2], joint_high[2], 9 * pi/180)],
                [p for p in np.arange(joint_low[3], joint_high[3], 12 * pi/180)],
                [0],
            ]
            self.workspace.make(self.robot, joint_positions, workspace_min_r, objs=self.obstacles)
            self.workspace.save(workspace_name)

        # Init node.
        rospy.init_node('core_controller_node')

        # Register services.
        self.__services = {}
        self.__register_service(ServiceLibrary.step_world, StepWorld)
        self.__register_service(ServiceLibrary.place_markers, PlaceMarkers)

        # Register publishers.
        self.__publishers = {}
        self.__register_publisher(TopicLibrary.joint0_com, Float64)
        self.__register_publisher(TopicLibrary.joint1_com, Float64)
        self.__register_publisher(TopicLibrary.joint2_com, Float64)
        self.__register_publisher(TopicLibrary.joint3_com, Float64)
        self.__register_publisher(TopicLibrary.joint4_com, Float64)

        # Register subscribers.
        self.__subscribers = {}
        self.__register_subscriber(TopicLibrary.joint_states, JointState)
        self.__register_subscriber(TopicLibrary.link1_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link2_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link3_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link4_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link_states, LinkStates)

        # Spin to listen to topic events.
        self.__spin_thread = SpinThread()
        self.__spin_thread.start()

        # Wait for topics.
        while True:
            self.__step_world()
            if not self.state() is None:
                break

    def __register_service(self, name: str, type: Type) -> None:
        if name in self.__services:
            raise Exception('Duplicated services')
        rospy.wait_for_service(name)
        self.__services[name] = rospy.ServiceProxy(name, type, persistent=True)

    def __register_publisher(self, name: str, type: Type) -> None:
        if name in self.__publishers:
            raise Exception('Duplicated publishers')
        self.__publishers[name] = rospy.Publisher(name, type, queue_size=10)

    def __register_subscriber(self, name: str, type: Type) -> None:
        if name in self.__subscribers:
            raise Exception('Duplicated subscribers')
        self.__subscribers[name] = None
        def callback(_name: str, _msg: Any) -> None:
            self.__subscribers[_name] = _msg
        callback_wrapper = functools.partial(callback, name)
        rospy.Subscriber(name, type, callback_wrapper)

    def __call_service(self, name: str, *args) -> Any:
        return self.__services[name](*args)

    def __get_publisher(self, name: str) -> Any:
        return self.__publishers[name]

    def __get_subscriber(self, name: str) -> Any:
        return self.__subscribers[name]

    def __step_world(self) -> None:
        step_iterations = self.configs.get(config.Environment.Gazebo.StepIterations_)
        self.__call_service(ServiceLibrary.step_world, step_iterations)
        self._state = None

    def __step(self, joint_position: np.ndarray) -> None:
        joint_position = self.robot.clip(joint_position)
        self.__get_publisher(TopicLibrary.joint0_com).publish(joint_position[0])
        self.__get_publisher(TopicLibrary.joint1_com).publish(joint_position[1])
        self.__get_publisher(TopicLibrary.joint2_com).publish(joint_position[2])
        self.__get_publisher(TopicLibrary.joint3_com).publish(joint_position[3])
        self.__get_publisher(TopicLibrary.joint4_com).publish(joint_position[4])

        # Handle controller/sensor delay
        i = 0
        while True:
            self.__step_world()
            state = self.state()
            if state.collision:
                break
            err = np.sum(np.abs(self.state().joint_position - joint_position))
            if err <= 1e-2:
                break
            i += 1
            if i > 30:
                raise Exception()

    def __random_state(self) -> None:
        while True:
            joint_pos = np.random.uniform(self.robot.joint_limits[0], self.robot.joint_limits[1])
            self.__step(joint_pos)
            state = self.state()
            if not state.collision:
                break

    def _get_state(self) -> GazeboState:
        joint_states: JointState = self.__get_subscriber(TopicLibrary.joint_states)
        link1_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link1_bumper)
        link2_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link2_bumper)
        link3_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link3_bumper)
        link4_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link4_bumper)
        link_states: LinkStates = self.__get_subscriber(TopicLibrary.link_states)

        if any([i is None for i in [
            joint_states,
            link1_bumper,
            link2_bumper,
            link3_bumper,
            link4_bumper,
            link_states,
        ]]): return None
        
        state = GazeboState()
        state.from_joint_states(joint_states)
        state.collision = (
            len(link1_bumper.states) +
            len(link2_bumper.states) +
            len(link3_bumper.states) +
            len(link4_bumper.states)
        ) > 0
        pos_achieved = link_states.pose[link_states.name.index('robot::effector')].position
        state.achieved = np.array([pos_achieved.x, pos_achieved.y, pos_achieved.z])
        state.desired = self.__desired
        return state

    def place_marker(self, marker: str, pos: np.ndarray) -> None:
        pt = Point()
        pt.x = pos[0]
        pt.y = pos[1]
        pt.z = pos[2]
        self.__call_service(ServiceLibrary.place_markers, marker, pt)
        
    def close(self) -> None:
        self.__spin_thread.terminate()
        Simulator.__client_activated = False

    def _reset(self) -> None:
        # Set target point randomly.
        self.__desired = self.workspace.sample()
        self.place_marker('marker_red', self.__desired)

        # Randomly reset robot state.
        self.__random_state()

    def _step(self, action: np.ndarray) -> None:
        action = action.clip(-1, 1)
        last_position = self.state().joint_position
        this_position = last_position + action * self.action_amp
        self.__step(this_position)
        state = self.state()
        if state.collision:
            self.__step(last_position)
            self.state().collision = True

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        pass

    def dim_action(self) -> int:
        return 5
        
    def reward(self) -> GazeboReward:
        return GazeboReward(self.robot, self.obstacles)