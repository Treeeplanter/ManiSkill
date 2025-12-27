"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Optional, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.agents.controllers.pd_ee_pose import PDEEPosController, PDEEPoseController
from mani_skill.agents.robots.panda.fr3_wristcam import FR3RobotiqWristCam


def build_solid_color_target(
    scene,
    radius: float,
    thickness: float,
    name: str,
    color: np.ndarray,
    body_type: str = "dynamic",
    add_collision: bool = False,
    initial_pose: Optional[sapien.Pose] = None,
):
    """
    build a solid color target
    
    Args:
        scene: ManiSkill scene object
        radius: target half size
        thickness: target thickness
        name: actor name
        color: RGBA color array, range [0, 1]
        body_type: "dynamic", "kinematic", or "static"
        add_collision: whether to add collision
        initial_pose: initial pose
    """
    builder = scene.create_actor_builder()
    
    # only create a solid color block
    builder.add_box_visual(
        half_size=[radius, radius, thickness / 2],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    if add_collision:
        builder.add_box_collision(half_size=[radius, radius, thickness / 2])
    
    # create actor based on body type
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    else:
        raise ValueError(f"Unknown body_type: {body_type}")
    
    if initial_pose is not None:
        actor.set_pose(initial_pose)
    
    return actor


# predefined color constants (for convenience)
COLORS = {
    'red': np.array([255, 0, 0, 255]) / 255,
    'green': np.array([0, 255, 0, 255]) / 255,
    'blue': np.array([0, 100, 255, 255]) / 255,
    'yellow': np.array([255, 255, 0, 255]) / 255,
    'purple': np.array([128, 0, 255, 255]) / 255,
    'orange': np.array([255, 165, 0, 255]) / 255,
    'cyan': np.array([0, 255, 255, 255]) / 255,
    'pink': np.array([255, 192, 203, 255]) / 255,
    'lime': np.array([50, 200, 50, 255]) / 255,
    'white': np.array([255, 255, 255, 255]) / 255,
    'black': np.array([0, 0, 0, 255]) / 255,
    'gray': np.array([128, 128, 128, 255]) / 255,
}

@register_env("PickPlace1Cube-v1", max_episode_steps=50)
class PickPlace1CubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to pick up a cube and place it on a goal region. there are two cubes on the table, the robot needs to pick up one of them and place it on the goal region.

    **Randomizations:**
    - the two cubes' xy position are randomized on top of a table in the region [-0.1, 0.0] x [-0.05, 0.05] and [0.0, 0.1] x [-0.05, 0.05]. It is placed flat on the table
    - the two cubes' z-axis rotation are randomized in [-pi/4, pi/4]

    **Success Conditions:**
    - one of the two cubes is placed on one of the goal regions
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fr3_robotiq_wristcam"]

    # Specify some supported robot types
    
    agent: Union[Panda, PandaWristCam, FR3RobotiqWristCam]

    # set some commonly used values
    goal_radius = 0.05
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="fr3_robotiq_wristcam", robot_init_qpos_noise=0.01, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )


    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        
        # 定义相机在机器人基座坐标系中的局部位姿
        droid_exo_left_local = sapien.Pose(
            p=[-0.12, 0.32, 0.6], 
            q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        )
        droid_exo_right_local = sapien.Pose(
            p=[-0.12, -0.32, 0.6], 
            q=[0.9622501868990583, -0.022557566113149838, 0.0841859828293692, 0.25783416049629954]
        )

        pose_base = sapien_utils.look_at([0.0, 0.0, 0.45], [-0.3, 0.0, 0.08])
        pose_steer_left = sapien_utils.look_at([-0.15, -0.35, 0.4], [-0.4, 0.0, -0.1])
        pose_steer_right = sapien_utils.look_at([-0.15, 0.35, 0.4], [-0.4, 0.0, -0.1])
        
        return [
            # CameraConfig(
            #     "base_camera",
            #     pose=droid_exo_left_local,
            #     width=256,
            #     height=256,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            # ),
            CameraConfig(
                uid="droid_exo_left_camera",
                pose=droid_exo_left_local,  # 直接使用局部位姿
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),
            CameraConfig(
                uid="droid_exo_right_camera",
                pose=droid_exo_right_local,  # 添加右相机
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.1,
                far=100,
                mount=self.agent.robot.links_map["base"],  # 挂载到机器人基座
            ),
            # CameraConfig(
            #     uid= "steer_left_camera",
            #     pose=pose_steer_left,
            #     width=512,
            #     height=512,
            #     fov=np.pi / 2,
            #     near=0.1,
            #     far=100,
            # ),
            # CameraConfig(
            #     uid= "steer_right_camera",
            #     pose=pose_steer_right,
            #     width=512,
            #     height=512,
            #     fov=np.pi / 2,
            #     near=0.1,
            #     far=100,
            # )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.25, 0.0, 0.25], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1.2, near=0.01, far=100
        )

        # droid_exo_left_local = sapien.Pose(
        #     p=[-0.12, 0.32, 0.6], 
        #     q=[0.9622501868990583, 0.022557566113149834, 0.08418598282936919, -0.25783416049629954]
        # )

        # return CameraConfig(
        #     "render_camera", pose=droid_exo_left_local, width=512, height=512, fov=np.pi/2, near=0.01, far=100, mount=self.agent.robot.links_map["base"]
        # )

    def _load_agent(self, options: dict):
        # set a reasonable initial pose for the agent that doesn't intersect other objects
        super()._load_agent(options, sapien.Pose(p=[0.0, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        # finally we specify an initial pose for the cube so that it doesn't collide with other objects initially
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([210, 10, 10, 255]) / 255,
            name="cube1",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place）
        self.goal_region1 = build_solid_color_target(
            scene=self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region1",
            color=COLORS['purple'],  
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )

        self.goal_region2 = build_solid_color_target(
            scene=self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region2",
            color=COLORS['lime'],  
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
        

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)
            
            # Override robot position after table_scene.initialize() sets it

            init_qpos = torch.tensor([[                   
                0.0,
                -1 /5 * np.pi,
                0,
                -np.pi * 4 / 5,
                0,
                np.pi * 3 / 5,
                0,
                0,
                0,
                0,
                0,
                0.04,
                0.04
            ]])
            self.agent.robot.set_qpos(init_qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.85, 0, -0.2]))

            # here we write some randomization code that randomizes the x, y position of the cubes
            # Cube: x in [0, 0.1], y in [-0.05, 0.05]
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = torch.rand(b) * 0.05 - 0.4
            xyz[..., 1] = torch.rand(b) * 0.05 - 0.15
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even use env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(obj_pose)


            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
            # and we further rotate 90 degrees on the y-axis to make the target object face up
            # set a little bit above 0 (1e-3) so the target is sitting on the table
            target_region_xyz = [
                torch.tensor([-0.5, 0.15, 1e-3]),
                torch.tensor([-0.2, 0.15, 1e-3]),
            ]

            self.goal_region1.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz[0],
                    q=euler2quat(0, 0, 0),
                )
            )
            self.goal_region2.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz[1],
                    q=euler2quat(0, 0, 0),
                )
            )

    def _get_steered_controller(self):
        controller = self.agent.controller
        if hasattr(controller, 'controllers') and isinstance(controller.controllers, dict):
            for ctrl in controller.controllers.values():
                if isinstance(ctrl, (PDEEPosController, PDEEPoseController)):
                    return ctrl
        return None
    
    def _is_obj_placed(self, obj: Actor, goal_region: Actor) -> torch.Tensor:
        return (
            torch.linalg.norm(
                obj.pose.p[..., :2] - goal_region.pose.p[..., :2], dim=-1
            )
            < self.goal_radius * 1.2
        ) & (obj.pose.p[..., 2] < self.cube_half_size + 5e-3)


    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position) and
        # the cube is still on the surface
        # success = self._is_obj_placed(self.cube, self.goal_region2) 
                # | self._is_obj_placed(self.cube, self.goal_region2) \
        success = self._is_obj_placed(self.cube, self.goal_region1) \
            | self._is_obj_placed(self.cube, self.goal_region2)
 
        
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs
