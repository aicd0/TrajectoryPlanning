import numpy as np
from simulator.ROS.game import Game
from simulator.ROS.game_state import GameState
from typing import Any, Type

# Import ROS packages.
import rospy
from gazebo_msgs.srv import GetJointProperties
from robot_sim.srv import StepWorld
# from std_msgs.msg import Float64

class ServiceLibrary:
    step_world = '/user/step_world'
    get_joint_properties = '/gazebo/get_joint_properties'

class TopicLibrary:
    link_states = '/gazebo/link_states'
    link1_bumper = '/link1_bumper'
    link2_bumper = '/link2_bumper'
    link3_bumper = '/link3_bumper'
    joint0_com = '/robot/joint0_position_controller/command'
    joint1_com = '/robot/joint1_position_controller/command'
    joint2_com = '/robot/joint2_position_controller/command'
    joint3_com = '/robot/joint3_position_controller/command'

class Simulator:
    __client_num = 0

    def __init__(self):
        # Init node.
        rospy.init_node('client_%d' % Simulator.__client_num)
        Simulator.__client_num += 1

        # Register services.
        self.__services = {}
        self.__register_service(ServiceLibrary.step_world, StepWorld)
        self.__register_service(ServiceLibrary.get_joint_properties, GetJointProperties)

        # Register publishers.
        self.__publishers = {}

        # Register subscribers.
        self.__subscribers = {}
        self.__register_subscriber

    def __register_service(self, name: str, type: Type) -> None:
        if name in self.__services:
            raise Exception('Duplicated services')
        rospy.wait_for_service(name)
        self.__services[name] = rospy.ServiceProxy(name, type)

    def __register_publisher(self, name: str, type: Type) -> Any:
        if name in self.__publishers:
            raise Exception('Duplicated publishers')
        self.__publishers[name] = rospy.Publisher(name, type, queue_size=10)

    def __register_subscriber(self, name: str, type: Type) -> None:
        if name in self.__subscribers:
            raise Exception('Duplicated subscribers')
        self.__subscribers[name] = None
        def callback(msg: Any) -> None:
            self.__subscribers[name] = msg
        rospy.Subscriber(name, type, callback)

    def __get_service(self, name: str) -> Any:
        return self.__services[name]

    def state(self) -> GameState:
        joint0 = self.__get_service(ServiceLibrary.get_joint_properties)('joint0')
        joint1 = self.__get_service(ServiceLibrary.get_joint_properties)('joint1')
        joint2 = self.__get_service(ServiceLibrary.get_joint_properties)('joint2')
        joint3 = self.__get_service(ServiceLibrary.get_joint_properties)('joint3')

        state = GameState()
        return state

    def close(self):
        pass

    def reset(self) -> GameState:
        # state_raw = self.env.reset()
        # game_state = GameState()
        # game_state.from_reset(state_raw)
        # return game_state
        pass

    def step(self, action: np.ndarray) -> GameState:
        # if self.action_discrete:
        #     action = np.clip(int((action[0] + 1) / 2 * self.n_action), 0, self.n_action - 1)

        # state, reward_raw, done, _ = self.env.step(action)
        # game_state = GameState()
        # game_state.from_step(state, reward_raw, done)
        # return game_state
        pass

    def plot_reset(self) -> None:
        pass

    def plot_step(self) -> None:
        pass

    def dim_action(self) -> int:
        # return self.__dim_action
        pass