"""
*********************************************************************************************************
*
*	模块名称 : carla 环境适配模块
*	文件名称 : carla_env.py
*	版    本 : V1.0.0
*   作    者 ：fc
*   日    期 ：2019/03/03
*	说    明 : OpenAI Gym compatible Driving simulation environment based on Carla.
              Requires the system environment variable CARLA_SERVER to be defined and be pointing to the
              CarlaUE4.sh file on your system. The default path is assumed to be at: ~/software/CARLA_0.8.2/CarlaUE4.sh.
*
* \par Method List:
*    1.    cleanup();
*    2.    print_measurements(measurements);
*    3.    check_collision(py_measurements);
*
* \par Instance Method List:
*   class CarlaEnv
*    1.    init_server();
*    2.    clear_server_state();
*    3.    reset();
*    4.    reset_env();
*    5.    encode_observation(image, py_measurements);
*    6.    step(action);
*    7.    step_env(action);
*    8.    preprocess_image(image);
*    9.    _read_observation();
*    10.   calculate_reward(current_measurement);
*
*   修订记录：
	2019-03-04：   1.0.0      build;
							  position_tasks设定为straight_poses_Town01
							  修改calculate_reward(), 按REWARD_ASSIGN_PARAMETERS中的超参数来计算reward,
							  将forward_speed按speed_limit归一化为[0,100]范围, forward_speed-->forward_speed_post_process
							  把超速情况考虑进calculate_reward()内, 但是没有考虑速度一直为0的情况

    2019-03-05:   1.0.0       解决Carla.sh一直吃内存问题
                              calculate_reward()仍然不合理，目前参数训练16W步，mean_100ep_reward在[25,29](最高奖励100)

    2019-03-11:   1.0.0       修改action_space为连续空间, discrete_actions=False;
                              注释preprocess_image()中对image做normalization的代码;

    2019-03-12:   1.0.0       修改action_space为4维;
                              framestack 改为4，采用deque存储, image格式改为np.uint8;

    2019-03-15:   1.0.0       ACTION_SPACE改为3;
                              定义REWARD_THRESHOLD_LOW = -100
                              
*	Copyright (C), 2015-2019, 阿波罗科技 www.apollorobot.cn
*
*********************************************************************************************************
"""
import os
import sys
import cv2
import random
import signal
import time
import json
import gym
import atexit
import subprocess
import traceback
import numpy as np
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque
from datetime import datetime
from gym.spaces import Box, Discrete, Tuple
sys.path.append("/home/fc/projects/carla-DRL/utils/")
from common import DEBUG_PRINT as DEBUG_PRINT

# Set this to the path to your Carla binary
SERVER_BINARY = os.environ.get("CARLA_SERVER", os.path.expanduser("~/toolkits/CARLA/carla-0.8.2/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"



# Import Carla python client API funcs
try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except ImportError:
    from .carla.client import CarlaClient
    from .carla.sensor import Camera
    from .carla.settings import CarlaSettings
    from .carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL: "REACH_GOAL",
    GO_STRAIGHT: "GO_STRAIGHT",
    TURN_RIGHT: "TURN_RIGHT",
    TURN_LEFT: "TURN_LEFT",
    LANE_FOLLOW: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

# Load scenario configuration parameters from scenarios.json
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
scenario_config = json.load(open(os.path.join(__location__, "scenarios.json")))
weathers = [scenario_config['Weather']['WetNoon'],
            scenario_config['Weather']['ClearSunset'],
            scenario_config['Weather']['MidRainyNoon'],
            scenario_config['Weather']['HardRainNoon'],
            scenario_config['Weather']['ClearSunset'],
            scenario_config['Weather']['CloudySunset']]
scenario_config['weather_distribution'] = weathers

# Default environment configuration
ENVIRONMENT_CONFIG = {
    "discrete_actions": False,
    "use_gray_or_depth_image":True,
    "use_image_only_observations": True,  # Exclude high-level planner inputs & goal info from the observations
    "server_map": "/Game/Maps/" + scenario_config["city"][0], # Town02
    "scenarios": scenario_config["Lane_Keep_Town1"], #[scenario_config["Lane_Keep_Town1"],scenario_config["Lane_Keep_Town2"]],
    "use_random_position_points": False,
    "framestack": 4,  # note: only [1, 2, 3,4] currently supported
    "enable_planner": True,
    "use_depth_camera": True,
    "early_terminate_on_collision": True,
    "verbose": False,
    "render" : True,  # Render to display if true
    "render_x_res": 800,
    "render_y_res": 600,
    "camera_position":(0.3, 0, 1.3),
    "image_x_res": 84,
    "image_y_res": 84,
    "seed": 1
}

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 10
# max speed km/h
MAX_SPEED_LIMIT = 30
# the threshold for total_reward
REWARD_THRESHOLD_LOW = -1.0
# Define the discrete action space
ACTION_DIMENSIONS = 3
THROTTLE_INDEX = 0
BRAKE_INDEX = 0
STEER_INDEX = 1
REVERSE_INDEX = 2
DISCRETE_ACTIONS = {
    #  throttle brake steer reverse
    0: [0.0, 0.0, 0.0],    # Coast
    1: [0.0, -0.5, 0.0],   # Turn Left
    2: [0.0, 0.5, 0.0],    # Turn Right
    3: [1.0, 0.0, 0.0],    # Forward
    4: [-0.5,0.0, 0.0],   # Brake
    5: [1.0, -0.5, 0.0],   # Bear Left & accelerate
    6: [1.0, 0.5, 0.0],    # Bear Right & accelerate
    7: [-0.5,-0.5, 0.0],  # Bear Left & decelerate
    8: [-0.5, 0.5, 0.0],   # Bear Right & decelerate
    # 9: [0.0, 0.0, -0.5, 1.0],  # Turn back Left
    # 10: [0.0, 0.0, 0.5, 1.0],  # Turn back Right
    # 11: [1.0, 0.0, 0.0, 1.0],  # Backward
    # 12: [1.0, 0.0, -0.5, 1.0],  # Bear back Left & accelerate
    # 13: [1.0, 0.0, 0.5, 1.0],  # Bear back Right & accelerate
    # 14: [0.0, 0.5, -0.5, 1.0],  # Bear back Left & decelerate
    # 15: [0.0, 0.5, 0.5, 1.0],  # Bear back Right & decelerate
}

# define the parameters used to assign the hyperparameters of several factors
# related to calculate the reward
REWARD_ASSIGN_PARAMETERS = {
    "location_coefficient": 1.0,
    "speed_coefficient": 0.05,
    "collision_coefficient": -0.00002,
    "offroad_coefficient": -2.0,
    "otherland_coefficient":-2.0
}

live_carla_processes = set()  # To keep track of all the Carla processes we launch to make the cleanup easier
def cleanup():
    """
    注册程序退出时的回调函数，在这个回调函数中做一些资源清理的操作
    如果程序是非正常crash，或者通过os._exit()退出，注册的回调函数将不会被调用
     """
    DEBUG_PRINT("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)
atexit.register(cleanup)

def print_measurements(measurements):
    """
     打印从Carla中获取的measurements

     :param measurements: the raw measurements data retrieve from read_data()
     :return: None
     """
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.2f}, {pos_y:.2f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,  # cm -> m
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    DEBUG_PRINT(message)

def check_collision(py_measurements):
    """
     检测任何collision发生或者total_reward < -100

     :param py_measurements: the player measurements data
     :return: bool
     """
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0 or m["collision_other"] > 0)
    return bool(collided)

class CarlaEnv(gym.Env):
    def __init__(self, config=ENVIRONMENT_CONFIG):
        """
        Carla Gym Environment class implementation. Creates an OpenAI Gym compatible driving environment based on
        Carla driving simulator.
        :param config: A dictionary with environment configuration keys and values
        """
        self.config = config
        self.config_city = self.config["server_map"].split("/")[-1]

        if self.config["enable_planner"]:
            self.config_planner = Planner(self.config_city)

        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(ACTION_DIMENSIONS,), dtype=np.float32)

        if config["use_gray_or_depth_image"]:
            image_space = Box(low=0, high=255,
                              shape=(config["image_y_res"], config["image_x_res"], 1 * config["framestack"]),
                              dtype=np.uint8)
        else:
            image_space = Box(low=0, high=255,
                              shape=(config["image_y_res"], config["image_x_res"], 3 * config["framestack"]),
                              dtype=np.uint8)

        if self.config["use_image_only_observations"]:
            self.observation_space = image_space
        else:
            self.observation_space = Tuple(
                [image_space,
                 Discrete(len(COMMANDS_ENUM)),  # next_command
                 Box(low=-128.0, high=128.0, shape=(2,), dtype=np.float32)])  # forward_speed, dist to goal

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._seed = ENVIRONMENT_CONFIG["seed"]

        self.server_port = None
        self.server_process = None
        self.client = None
        self.global_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.images = deque([], maxlen=config["framestack"])
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None

    def init_server(self):
        '''
               启动carla仿真环境
        :return:
        '''
        DEBUG_PRINT("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = random.randint(10000,60000)

        if self.config["render"]:
            self.server_process = subprocess.Popen(
                                                    "{} {} -windowed -ResX={} -ResY={} -carla-server -carla-world-port={}".format(
                                                            SERVER_BINARY,
                                                            self.config["server_map"],
                                                            self.config["render_x_res"],
                                                            self.config["render_y_res"],
                                                            self.server_port),
                                                     shell=True,
                                                     preexec_fn=os.setsid,
                                                     stdout=open(os.devnull, "w"))
        else:
            self.server_process = subprocess.Popen(
                ("SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} {} " +
                 self.config["server_map"] + " -windowed -ResX={} -ResY={} -carla-server -carla-world-port={}").
                 format(0, SERVER_BINARY, self.config["render_x_res"], self.config["render_y_res"], self.server_port),
                 shell=True, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))

        live_carla_processes.add(os.getpgid(self.server_process.pid))

        for i in range(RETRIES_ON_ERROR):
            try:
                self.client = CarlaClient("localhost", self.server_port)
                return self.client.connect()
            except Exception as e:
                DEBUG_PRINT("Error connecting: {}, attempt {}".format(e, i))
                time.sleep(2)

    def clear_server_state(self):
        '''
                断开client连接, 清除进程
        :return:
        '''
        DEBUG_PRINT("Clearing Carla server state")
        try:
            if self.client:
                self.client.disconnect()
                self.client = None
        except Exception as e:
            DEBUG_PRINT("Error disconnecting client: {}".format(e))
            pass
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None
            self.images.clear()

    def __del__(self):
        self.clear_server_state()

    def reset(self):
        error = None
        self.clear_server_state()
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self.reset_env()
            except Exception as e:
                DEBUG_PRINT("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    def reset_env(self):
        self.global_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # 对Carla环境进行配置
        settings = CarlaSettings()
        # 如果是单个场景（地图），则用此地图；否则随机选择一个场景
        if isinstance(self.config["scenarios"], dict):
            self.scenario = self.config["scenarios"]
        else:
            self.scenario = random.choice(self.config["scenarios"])

        assert self.scenario["city"] == self.config_city, (self.scenario, self.config_city)

        # 随机选择weather
        self.weather = random.choice(self.scenario["weather_distribution"])
        settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=self.scenario["num_vehicles"],
                    NumberOfPedestrians=self.scenario["num_pedestrians"],
                    WeatherId=self.weather)
        settings.randomize_seeds()

        # 在正前方安装摄像头
        if self.config["use_depth_camera"]:
            camera1 = Camera("CameraDepth", PostProcessing="Depth")
            camera1.set_image_size(self.config["render_x_res"], self.config["render_y_res"])
            camera1.set_position(*(self.config['camera_position']))
            settings.add_sensor(camera1)

        camera2 = Camera("CameraRGB")
        camera2.set_image_size(self.config["render_x_res"], self.config["render_y_res"])
        camera2.set_position(*(self.config['camera_position']))
        settings.add_sensor(camera2)

        # Now we load these settings into the server. The server replies
        # with a scene description containing the available start spots for the player.
        scene = self.client.load_settings(settings)
        positions = scene.player_start_spots
        # 读取配置文件中的start_pos, end_pos
        if self.config["use_random_position_points"]:
            straight_poses_tasks_len = len(scenario_config["Straight_Poses_Town01"])
            random_position_task = scenario_config["Straight_Poses_Town01"][np.random.randint(straight_poses_tasks_len)]
            self.scenario["start_pos_id"] = random_position_task[0]
            self.scenario["end_pos_id"] = random_position_task[1]
        else:
            pass
        self.start_pos = positions[self.scenario["start_pos_id"]]
        self.end_pos = positions[self.scenario["end_pos_id"]]
        self.start_coord = [self.start_pos.location.x, self.start_pos.location.y]
        self.end_coord = [self.end_pos.location.x, self.end_pos.location.y]
        DEBUG_PRINT( "Start pos {} ({}), end {} ({})".format(
                    self.scenario["start_pos_id"], self.start_coord,
                    self.scenario["end_pos_id"], self.end_coord))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        DEBUG_PRINT("Starting new episode...")
        self.client.start_episode(self.scenario["start_pos_id"])
        image, py_measurements = self._read_observation()
        self.prev_measurement = py_measurements

        return self.encode_observation(self.preprocess_image(image), py_measurements)

    def encode_observation(self, image, py_measurements):
        '''
                stack raw image and measurments data
        :param image:
        :param py_measurements:
        :return:
        '''
        assert self.config["framestack"] in [1, 2, 3, 4]

        while len(self.images) < self.config["framestack"]:
            self.images.append(image)
        # 将每一帧存入frame队列
        self.images.append(image)
        images = np.array(self.images)
        shape0, shape1, shape2 = images.shape
        images = images.reshape(shape1,shape2,shape0)


        if self.config["use_image_only_observations"]:
            obs = images
        else:
            obs = (
                images,
                COMMAND_ORDINAL[py_measurements["next_command"]],
                [py_measurements["agent_forward_speed"], py_measurements["distance_to_goal"]])
        self.last_obs = obs
        return obs

    def step(self, action):
        try:
            obs = self.step_env(action)
            return obs
        except Exception:
            DEBUG_PRINT("Error during step, terminating episode early", traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})

    def step_env(self, action):
        '''
                更新环境: 发送控制信息给环境, 读取传感器数据,计算奖励,返回observation
        :param action:
        :return:
        '''
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]
            assert len(action) == ACTION_DIMENSIONS, "Invalid action {}".format(action)

        # 将离散动作映射到 throttle,brake, steer, 暂时不适用reverse,hand_brake
        throttle = float(np.clip(action[THROTTLE_INDEX], 0, 1))
        brake = float(abs(np.clip(action[BRAKE_INDEX], -1, 0)))
        steer = float(np.clip(action[STEER_INDEX], -1, 1))
        reverse = bool(np.clip(action[REVERSE_INDEX], 0, 1) >= 0.5)
        hand_brake = False

        # if self.config["verbose"]:
        #     DEBUG_PRINT("steer = ", steer, " throttle =", throttle, " brake = ", brake, " reverse = ", reverse)

        self.client.send_control( steer=steer, throttle=throttle, brake=brake,
                                    hand_brake=hand_brake, reverse=reverse)

        # Process observations
        image, py_measurements = self._read_observation()

        # if self.config["verbose"]:
        DEBUG_PRINT("Next command", py_measurements["next_command"])

        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action

        py_measurements["control"] = {"steer": steer,
                                      "throttle": throttle,
                                      "brake": brake,
                                      "reverse": reverse,
                                      "hand_brake": hand_brake}

        reward = self.calculate_reward(py_measurements)
        
        self.total_reward += reward

        # if self.config["verbose"]:
        DEBUG_PRINT("Current total reward {:+.2f}".format(self.total_reward))

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        
        done = ( self.global_steps > self.scenario["max_steps"] or
                 py_measurements["next_command"] == "REACH_GOAL" or
                 self.total_reward < REWARD_THRESHOLD_LOW or
                (self.config["early_terminate_on_collision"] and check_collision(py_measurements)))
        
        py_measurements["done"] = done
        self.prev_measurement = py_measurements

        self.global_steps += 1
        image = self.preprocess_image(image)

        return (self.encode_observation(image, py_measurements), reward, done, py_measurements)

    def preprocess_image(self, raw_image):
        '''
                处理raw image
        :param image:
        :return:
        '''
        # 将image.data 范围[0, 255]
        if self.config["use_depth_camera"]:
            # raw depth image 范围[0,1]
            data = raw_image.data * 255
            # DEBUG_PRINT("image shape: ", data.shape)
            # 因为图片保存时x,y倒置
            data = data.reshape(self.config["render_y_res"], self.config["render_x_res"], 1)
            data = cv2.resize(data, (self.config["image_x_res"], self.config["image_y_res"]), interpolation=cv2.INTER_AREA) #shrink the image

        else:
            data = raw_image.data.reshape(self.config["render_y_res"], self.config["render_x_res"], 3)

            if self.config["use_gray_or_depth_image"]:
                data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

            data = cv2.resize(data, (self.config["image_x_res"], self.config["image_y_res"]),interpolation=cv2.INTER_AREA)

        data = data.astype(np.uint8)

        # mean, std = data.mean(), data.std()
        # # DEBUG_PRINT("image mean: {:.2f}, image std: {:.2f}".format(mean, std))
        # if np.isnan(std):
        #     pass
        # else:
        #     data = (data - mean) / (std + 0.00001)

        return data

    def _read_observation(self):
        '''
        Read the data produced by the server this frame
        :return: image and measurements
        '''

        measurements, sensor_data = self.client.read_data()

        # Print some of the measurements.
        if self.config["verbose"]:
            print_measurements(measurements)

        observation = None
        if self.config["use_depth_camera"]:
            camera_name = "CameraDepth"
        else:
            camera_name = "CameraRGB"
        for name, image in sensor_data.items():
            if name == camera_name:
                observation = image

        current_measurement = measurements.player_measurements

        if self.config["enable_planner"]:
            next_command = COMMANDS_ENUM[
                self.config_planner.get_next_command(
                    [current_measurement.transform.location.x, current_measurement.transform.location.y, current_measurement.transform.location.z],
                    [current_measurement.transform.orientation.x, current_measurement.transform.orientation.y, current_measurement.transform.orientation.z],
                    [self.end_pos.location.x, self.end_pos.location.y, self.end_pos.location.z],
                    [self.end_pos.orientation.x, self.end_pos.orientation.y, self.end_pos.orientation.z])
            ]
        else:
            next_command = "LANE_FOLLOW"

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif self.config["enable_planner"]:
            distance_to_goal = self.config_planner.get_shortest_path_distance(
                [current_measurement.transform.location.x, current_measurement.transform.location.y, current_measurement.transform.location.z],
                [current_measurement.transform.orientation.x, current_measurement.transform.orientation.y, current_measurement.transform.orientation.z],
                [self.end_pos.location.x, self.end_pos.location.y, self.end_pos.location.z],
                [self.end_pos.orientation.x, self.end_pos.orientation.y, self.end_pos.orientation.z])
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(np.linalg.norm(
            [current_measurement.transform.location.x - self.end_pos.location.x,
             current_measurement.transform.location.y - self.end_pos.location.y]))

        result = {
            "episode_id": self.episode_id,
            "global_step": self.global_steps,
            "agent_location_x": current_measurement.transform.location.x,
            "agent_location_y": current_measurement.transform.location.y,
            "agent_orientation_x": current_measurement.transform.orientation.x,
            "agent_orientation_y": current_measurement.transform.orientation.y,
            "agent_forward_speed": current_measurement.forward_speed,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": current_measurement.collision_vehicles,
            "collision_pedestrians": current_measurement.collision_pedestrians,
            "collision_other": current_measurement.collision_other,
            "intersection_offroad": current_measurement.intersection_offroad,
            "intersection_otherlane": current_measurement.intersection_otherlane,
            "weather": self.weather,
            "map": self.config["server_map"],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
            "current_scenario": self.scenario,
            "image_x_res": self.config["image_x_res"],
            "image_y_res": self.config["image_y_res"],
            "num_vehicles": self.scenario["num_vehicles"],
            "num_pedestrians": self.scenario["num_pedestrians"],
            "max_steps": self.scenario["max_steps"],
            "next_command": next_command,
            "is_overspeed": (current_measurement.forward_speed > MAX_SPEED_LIMIT)
        }


        assert observation is not None, sensor_data
        return observation, result

    def calculate_reward(self, current_measurement):
        """
        Calculate the reward based on the effect of the action taken using the previous and the current measurements
        reward = delta(distance_to_goal)/100 + 0.05*delta(forward_speed) - 0.0002*delta(collision_*) - 2*delta(intersection_*)
        :param current_measurement: The measurement obtained from the Carla engine after executing the current action
        :return: The scalar reward
        """
        reward = 0.0

        cur_dist = current_measurement["distance_to_goal"]

        prev_dist = self.prev_measurement["distance_to_goal"]

        # if self.config["verbose"]:
        #     DEBUG_PRINT("Current distance to goal {}, Previous distance to goal {}".format(cur_dist, prev_dist))

        # Distance travelled toward the goal in m
        # constarined into [-100, +100]
        reward += REWARD_ASSIGN_PARAMETERS["location_coefficient"] * np.clip((prev_dist - cur_dist)/100, -10.0, 10.0)

        # Change in speed (km/hr)
        # limit speed less than 30km/h
        # 如果超速了，则把reward换做负号
        # if current_measurement["is_overspeed"]:
        #     reward -= REWARD_ASSIGN_PARAMETERS["speed_coefficient"] * (
        #             current_measurement["agent_forward_speed"] - self.prev_measurement["agent_forward_speed"])
        # else:
        #     reward += REWARD_ASSIGN_PARAMETERS["speed_coefficient"] * (
        #             current_measurement["agent_forward_speed"] - self.prev_measurement["agent_forward_speed"])
        reward += REWARD_ASSIGN_PARAMETERS["speed_coefficient"] * (
             current_measurement["agent_forward_speed"] - self.prev_measurement["agent_forward_speed"])

        # New collision damage
        reward += REWARD_ASSIGN_PARAMETERS["collision_coefficient"] * (
            current_measurement["collision_vehicles"] + current_measurement["collision_pedestrians"] + current_measurement["collision_other"]
           - self.prev_measurement["collision_vehicles"]-self.prev_measurement["collision_pedestrians"]-self.prev_measurement["collision_other"])

        # New sidewalk intersection
        reward += REWARD_ASSIGN_PARAMETERS["offroad_coefficient"] * (
            current_measurement["intersection_offroad"] - self.prev_measurement["intersection_offroad"])

        # New opposite lane intersection
        reward += REWARD_ASSIGN_PARAMETERS["otherland_coefficient"] * (
            current_measurement["intersection_otherlane"] - self.prev_measurement["intersection_otherlane"])

        return reward


import tensorflow as tf
import tensorflow.contrib.layers as layers

# if __name__ == "__main__":
#
#     session = tf.Session()
#
#     def model(inpt, num_actions, scope, reuse=False):
#         """This model takes as input an observation and returns values of all actions.
#         Input: RGB:[2, 224,224,3] (NHWC)
#                DEPTH:[2, 224,224,1] (NHWC)
#         """
#         with tf.variable_scope(scope, reuse=reuse):
#             out = inpt
#             out = layers.conv2d(inpt, 32, kernel_size=[11, 11], stride=4, padding='SAME',activation_fn=tf.nn.relu)  # normalizer_fn=tf.nn.batch_normalization)
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.conv2d(out, 64, kernel_size=[5, 5], stride=2, padding='SAME', activation_fn=tf.nn.relu)
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.max_pool2d(out, kernel_size=[3, 3], stride=2, padding='VALID')
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.conv2d(out, 192, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu)
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#             DEBUG_PRINT(out.op.name, out.get_shape().as_list())
#             out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
#             return out
#
#     image = tf.Variable(tf.zeros(shape=[2,224,224,3]),dtype=tf.float32)
#     session.run(tf.global_variables_initializer())
#     for _ in range(5):
#         env = CarlaEnv()
#         obs = env.reset()
#         done = False
#         t = 0
#         total_reward = 0.0
#         DEBUG_PRINT("observation shape:",obs.shape)
#
#         # model(image.assign(obs.reshape([2,224,224,3])), env.action_space.n, scope='q_function')
#         while not done:
#             t += 1
#             if ENVIRONMENT_CONFIG["discrete_actions"]:
#                 obs, reward, done, info = env.step(3)  # Go Forward
#             else:
#                 obs, reward, done, info = env.step([1.0, 0.0])  # Full throttle, zero steering angle
#             total_reward += reward
#
#
#
#             DEBUG_PRINT("step#:", t, "reward:", round(reward, 4), "total_reward:", round(total_reward, 4), "done:", done)
