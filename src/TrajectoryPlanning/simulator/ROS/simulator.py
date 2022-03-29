import config
import functools
import numpy as np
import threading
import time
import utils.math
from simulator.ROS.game import Game
from simulator.ROS.game_state import GameState
from typing import Any, Type

# Import ROS packages.
import rospy
from gazebo_msgs.msg import ContactsState, LinkStates
from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesResponse
from robot_sim.srv import StepWorld
from std_msgs.msg import Float64

class SpinThread (threading.Thread):
    def run(self):
        if not rospy.core.is_initialized():
            raise rospy.exceptions.ROSInitException("client code must call rospy.init_node() first")
        self.__terminate_requested = False
        try:
            while not rospy.core.is_shutdown() and not self.__terminate_requested:
                rospy.rostime.wallsleep(0.5)
        except KeyboardInterrupt:
            rospy.core.signal_shutdown('keyboard interrupt')
    
    def terminate(self) -> None:
        self.__terminate_requested = True
        while self.is_alive():
            time.sleep(0.5)

class ServiceLibrary:
    step_world = '/user/step_world'
    place_target = '/user/place_target'
    get_joint_properties = '/gazebo/get_joint_properties'

class TopicLibrary:
    joint0_com = '/robot/joint0_position_controller/command'
    joint1_com = '/robot/joint1_position_controller/command'
    joint2_com = '/robot/joint2_position_controller/command'
    joint3_com = '/robot/joint3_position_controller/command'
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

        # Init node.
        rospy.init_node('core_controller_node')

        # Register services.
        self.__services = {}
        self.__register_service(ServiceLibrary.step_world, StepWorld)
        self.__register_service(ServiceLibrary.step_world, StepWorld)
        self.__register_service(ServiceLibrary.get_joint_properties, GetJointProperties)

        # Register publishers.
        self.__publishers = {}
        self.__register_publisher(TopicLibrary.joint0_com, Float64)
        self.__register_publisher(TopicLibrary.joint1_com, Float64)
        self.__register_publisher(TopicLibrary.joint2_com, Float64)
        self.__register_publisher(TopicLibrary.joint3_com, Float64)

        # Register subscribers.
        self.__subscribers = {}
        self.__register_subscriber(TopicLibrary.link1_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link2_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link3_bumper, ContactsState)
        self.__register_subscriber(TopicLibrary.link_states, LinkStates)

        # Spin to listen to topic events.
        self.__spin_thread = SpinThread()
        self.__spin_thread.start()

    def __register_service(self, name: str, type: Type) -> None:
        if name in self.__services:
            raise Exception('Duplicated services')
        rospy.wait_for_service(name)
        self.__services[name] = rospy.ServiceProxy(name, type)

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

    def __get_service(self, name: str) -> Any:
        return self.__services[name]

    def __get_publisher(self, name: str) -> Any:
        return self.__publishers[name]

    def __get_subscriber(self, name: str) -> Any:
        return self.__subscribers[name]

    def __state(self) -> GameState:
        joint0: GetJointPropertiesResponse = self.__get_service(ServiceLibrary.get_joint_properties)('joint0')
        joint1: GetJointPropertiesResponse = self.__get_service(ServiceLibrary.get_joint_properties)('joint1')
        joint2: GetJointPropertiesResponse = self.__get_service(ServiceLibrary.get_joint_properties)('joint2')
        joint3: GetJointPropertiesResponse = self.__get_service(ServiceLibrary.get_joint_properties)('joint3')
        link1_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link1_bumper)
        link2_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link2_bumper)
        link3_bumper: ContactsState = self.__get_subscriber(TopicLibrary.link3_bumper)
        link_states: LinkStates = self.__get_subscriber(TopicLibrary.link_states)
        
        state = GameState()
        state.joint_states = np.array([
            joint0.position[0],
            joint1.position[0],
            joint2.position[0],
            joint3.position[0],
        ])
        state.collision = len(link1_bumper.states) + len(link2_bumper.states) + len(link3_bumper.states) > 0
        pos_achieved = link_states.pose[link_states.name.index('robot::effector')].position
        state.achieved = np.array([pos_achieved.x, pos_achieved.y, pos_achieved.z])
        state.desired = self.__desired
        return state

    def close(self):
        self.__spin_thread.terminate()
        Simulator.__client_activated = False

    def reset(self) -> GameState:
        self.__desired = utils.math.random_point_in_hypersphere(3, low=0.2, high=2.0)
        if not self.__state_initialized:
            self.__get_service(ServiceLibrary.step_world)(config.Simulator.ROS.StepIterations)
            self.__state_initialized = True
        return self.__state()

    def step(self, action: np.ndarray) -> GameState:
        joint0_pos = action[0]
        joint1_pos = action[1]
        joint2_pos = action[2]
        joint3_pos = action[3]
        self.__get_publisher(TopicLibrary.joint0_com).publish(joint0_pos)
        self.__get_publisher(TopicLibrary.joint1_com).publish(joint1_pos)
        self.__get_publisher(TopicLibrary.joint2_com).publish(joint2_pos)
        self.__get_publisher(TopicLibrary.joint3_com).publish(joint3_pos)
        self.__get_service(ServiceLibrary.step_world)(config.Simulator.ROS.StepIterations)
        return self.__state()

    def plot_reset(self) -> None:
        raise NotImplementedError()

    def plot_step(self) -> None:
        pass

    def dim_action(self) -> int:
        return 4