import config
import functools
import numpy as np
import random
import sys
import threading
import time
import utils.platform
import utils.string_utils
from .game_state import GameState
from framework.configuration import global_configs as configs
from math import pi
from typing import Any, Type

# Import ROS and Gazebo packages.
if utils.platform.is_windows():
    sys.path.append(config.Environment.ROS.ROSLibPath)
    sys.path.append(config.Environment.ROS.ProjectLibPath)
import rospy
from gazebo_msgs.msg import ContactsState, LinkStates
from geometry_msgs.msg import Point
from robot_sim.srv import PlaceMarkers, StepWorld
from rospy.service import ServiceException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class JointLimit:
    class Joint0:
        L = -pi
        H = pi
    class Joint1:
        L = 0
        H = pi/2
    class Joint2:
        L = 0
        H = pi/2
    class Joint3:
        L = 0
        H = pi/2

class SpinThread (threading.Thread):
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
    joint_states = '/robot/joint_states'
    link1_bumper = '/link1_bumper'
    link2_bumper = '/link2_bumper'
    link3_bumper = '/link3_bumper'
    link_states = '/gazebo/link_states'

class Simulator:
    __client_activated = False

    def __init__(self):
        if Simulator.__client_activated:
            raise Exception()
        Simulator.__client_activated = True
        self.__state = None
        self.__desired = None

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

        # Register subscribers.
        self.__subscribers = {}
        self.__register_subscriber(TopicLibrary.joint_states, JointState)
        self.__register_subscriber(TopicLibrary.link1_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link2_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link3_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link_states, LinkStates)

        # Spin to listen to topic events.
        self.__spin_thread = SpinThread()
        self.__spin_thread.start()
        time.sleep(1)

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
        step_iterations = configs.get(config.Environment.ROS.StepIterations_)
        self.__call_service(ServiceLibrary.step_world, step_iterations)
        self.__state = None

    def __get_state(self) -> GameState:
        if self.__state is None:
            joint_states: JointState = self.__get_subscriber(TopicLibrary.joint_states)
            link1_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link1_bumper)
            link2_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link2_bumper)
            link3_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link3_bumper)
            link_states: LinkStates = self.__get_subscriber(TopicLibrary.link_states)
            
            self.__state = GameState()
            self.__state.from_joint_states(joint_states)
            self.__state.collision = len(link1_bumper.states) + len(link2_bumper.states) + len(link3_bumper.states) > 0
            pos_achieved = link_states.pose[link_states.name.index('robot::effector')].position
            self.__state.achieved = np.array([pos_achieved.x, pos_achieved.y, pos_achieved.z])
            self.__state.desired = self.__desired
            
        return self.__state

    def __step(self, joint_position: np.ndarray) -> None:
        self.__get_publisher(TopicLibrary.joint0_com).publish(joint_position[0])
        self.__get_publisher(TopicLibrary.joint1_com).publish(joint_position[1])
        self.__get_publisher(TopicLibrary.joint2_com).publish(joint_position[2])
        self.__get_publisher(TopicLibrary.joint3_com).publish(joint_position[3])
        self.__step_world()

    def __random_state(self) -> None:
        while True:
            random_joint_position = np.array([
                random.uniform(JointLimit.Joint0.L, JointLimit.Joint0.H),
                random.uniform(JointLimit.Joint1.L, JointLimit.Joint1.H),
                random.uniform(JointLimit.Joint2.L, JointLimit.Joint2.H),
                random.uniform(JointLimit.Joint3.L, JointLimit.Joint3.H),
            ])
            self.__step(random_joint_position)
            random_state = self.__get_state()
            if not random_state.collision:
                break
        
    def close(self) -> None:
        self.__spin_thread.terminate()
        Simulator.__client_activated = False

    def reset(self) -> GameState:
        # Generate target point randomly.
        self.__random_state()
        self.__desired = self.__get_state().achieved

        # Notify Gazebo to update target point.
        pt = Point()
        pt.x = self.__desired[0]
        pt.y = self.__desired[1]
        pt.z = self.__desired[2]
        self.__call_service(ServiceLibrary.place_markers, 'marker_red', pt)

        # Randomly initialize robot.
        self.__random_state()
        return self.__get_state()

    def step(self, action: np.ndarray) -> GameState:
        action_amp = configs.get(config.Environment.ROS.ActionAmp_)
        last_position = self.__get_state().joint_position
        this_position = last_position + action * action_amp
        self.__step(this_position)
        new_state = self.__get_state()
        if new_state.collision:
            self.__step(last_position)
            new_state = self.__get_state()
            new_state.collision = True
        return new_state

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        pass

    def dim_action(self) -> int:
        return 4