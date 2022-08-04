import gym
from gym.envs.registration import register

ENV_ID = {
    'navigation1': 'Navigation-v0',
    'navigation2': 'Navigation-v1',
    'maze': 'Maze-v0',
    'image_maze': 'ImageMaze-v0',
    'obj_extraction': 'ObjExtraction-v0',
    'obj_dynamic_extraction': 'ObjDynamicExtraction-v0',
    'drone_hover': 'DroneHover-v0',
    'kine_car': 'KineCar-v0',
    'drone_xz': 'Drone2d-v0',
    'kine_car_gazebo': 'KineCarGazebo-v0'
}

ENV_CLASS = {
    'navigation1': 'Navigation1',
    'navigation2': 'Navigation2',
    'maze': 'MazeNavigation',
    'image_maze': 'MazeImageNavigation',
    'obj_extraction': 'ObjExtraction',
    'obj_dynamic_extraction': 'ObjDynamicExtraction',
    'drone_hover': 'DroneHover',
    'kine_car': 'KineCar',
    'drone_xz': 'DroneXZ',
    'kine_car_gazebo': 'KineCarGazebo'
}

def make(env_name, render = False):
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point='env.' + env_name + ":" + env_class)
    return gym.make(env_id, render=render)

if __name__ == '__main__':
    # env = make('kine_car_gazebo')
    env = make('kine_car_gazebo')
    obs = env.reset()
    print(obs)