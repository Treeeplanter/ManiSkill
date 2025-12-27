#!/usr/bin/env python3

import argparse
import copy
from numpy import ndarray
import gymnasium as gym
import numpy as np
import torch
import h5py
import json
from dataclasses import dataclass
from typing import Annotated, Any
import tyro
from tqdm import tqdm

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.logging_utils import logger
import mani_skill.trajectory.utils as trajectory_utils

# GELLO imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig

# Import custom environments to trigger registration
from mani_skill.envs.tasks.steer_tasks.pickplace_1_cube import PickPlace1CubeEnv


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickPlace1Cube-v1"
    obs_mode: str = "rgb"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "fr3_robotiq_wristcam"
    """机器人类型，必须是 panda 系列"""
    record_dir: str = "demos"
    use_timestamp: bool = True
    """是否在文件名中添加时间戳（避免覆盖旧数据）"""
    save_video: bool = False
    """是否保存视频"""
    viewer_shader: str = "rt-fast"
    """查看器的 shader"""
    video_saving_shader: str = "rt-fast"
    """保存视频的 shader"""
    control_freq: float = 50.0
    """控制频率 (Hz)"""
    record_freq: float | None = 10.0
    """记录频率 (Hz)。如果为 None, 则与 control_freq 相同（每步都记录）"""
    max_episode_steps: int = 50000
    """每个 episode 的最大步数"""
    target_trajs: int = 200
    """目标轨迹数量"""
    # GELLO 
    gello_port: str = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0"
    """GELLO 设备的 USB 端口"""

class RecordEpisodeWithFreq(RecordEpisode):
    """
    继承 RecordEpisode，支持记录频率控制，保存 joint_pos actions（8维）。
    
    用于：用 pd_joint_pos 控制，保存 joint_pos actions，但支持采样（record_freq < control_freq）。
    
    Args:
        record_freq: 记录频率（Hz）。如果 None，每步都记录。
        control_freq: 控制频率（Hz）。
        其他参数同 RecordEpisode。
    """
    
    def __init__(self, *args, record_freq=None, control_freq=50.0, **kwargs):
        # 设置 save_on_reset=False，手动控制保存
        kwargs['save_on_reset'] = kwargs.get('save_on_reset', False)
        super().__init__(*args, **kwargs)
        self.record_freq = record_freq
        self.control_freq = control_freq
        self._step_count = 0  # 用于记录频率控制
        self._true_elapsed_steps = {}  # 存储每个环境的真实执行步数
        self._episode_true_steps = []  # 存储每个 episode 的真实步数
    
    def reset(self, *args, **kwargs):
        # 如果 save_on_reset=True，先处理之前的轨迹
        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            if self._trajectory_buffer is not None:
                options = kwargs.get('options')
                if options is None or "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
                else:
                    self.flush_trajectory(env_idxs_to_flush=common.to_numpy(options["env_idx"]))
                self._trajectory_buffer = None
        
        # 调用父类 reset（标准 RecordEpisode 逻辑，会初始化 buffer）
        obs, info = super().reset(*args, **kwargs)
        
        if info["reconfigure"]:
            self._trajectory_buffer = None
        
        # 重置步数计数器
        self._step_count = 0
        self._true_elapsed_steps[0] = 0  # 单环境情况
        
        return obs, info
    
    def step(self, action):
        """
        Args:
            action: joint_pos action [7 joints + 1 gripper] (8维)
        
        重写以支持记录频率控制，保存原始 joint_pos actions
        """
        # 执行环境 step
        if self.save_video and self._video_steps == 0:
            self.render_images.append(self.capture_image())
        
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        # 更新步数计数器（先更新，用于采样判断）
        self._step_count += 1
        self._true_elapsed_steps[0] = self._true_elapsed_steps.get(0, 0) + 1
        
        # 判断是否应该记录这一步
        should_record = True
        if self.record_freq is not None and self.record_freq < self.control_freq:
            record_interval = int(self.control_freq / self.record_freq)
            should_record = (self._step_count % record_interval == 0)
        
        # ✅ 关键修复：如果 success 发生，强制记录这一帧！
        if info.get("success", False):
            should_record = True
        
        # 保存 obs/state/action 到 buffer（仅在需要时）
        if self.save_trajectory and should_record and self._trajectory_buffer is not None:
            state_dict = self.base_env.get_state_dict()
            if self.record_env_state:
                self._trajectory_buffer.state = common.append_dict_array(
                    self._trajectory_buffer.state,
                    common.to_numpy(common.batch(state_dict)),
                )
            self._trajectory_buffer.observation = common.append_dict_array(
                self._trajectory_buffer.observation,
                common.to_numpy(common.batch(obs)),
            )
            # ✅ 保存原始 joint_pos action (8维)
            self._trajectory_buffer.action = common.append_dict_array(
                self._trajectory_buffer.action,
                common.to_numpy(common.batch(action)),
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.append_dict_array(
                    self._trajectory_buffer.reward,
                    common.to_numpy(common.batch(rew)),
                )
            self._trajectory_buffer.terminated = common.append_dict_array(
                self._trajectory_buffer.terminated,
                common.to_numpy(common.batch(terminated)),
            )
            self._trajectory_buffer.truncated = common.append_dict_array(
                self._trajectory_buffer.truncated,
                common.to_numpy(common.batch(truncated)),
            )
            done = terminated | truncated
            self._trajectory_buffer.done = common.append_dict_array(
                self._trajectory_buffer.done,
                common.to_numpy(common.batch(done)),
            )
            if "success" in info:
                self._trajectory_buffer.success = common.append_dict_array(
                    self._trajectory_buffer.success,
                    common.to_numpy(common.batch(info["success"])),
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = common.append_dict_array(
                    self._trajectory_buffer.fail,
                    common.to_numpy(common.batch(info["fail"])),
                )
            else:
                self._trajectory_buffer.fail = None

        # 处理视频
        if self.save_video:
            self._video_steps += 1
            if self.info_on_video:
                scalar_info = gym_utils.extract_scalars_from_info(
                    common.to_numpy(info), batch_size=self.num_envs
                )
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [float(r) for r in scalar_info["reward"]]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
                image = self.capture_image(scalar_info)
            else:
                image = self.capture_image()
            self.render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()
        
        self._elapsed_record_steps += 1
        
        return obs, rew, terminated, truncated, info
    
    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
    ):
        """
        重写以保存真实执行步数，然后调用父类 flush
        """
        # 保存真实执行步数
        if 0 in self._true_elapsed_steps:
            self._episode_true_steps.append(self._true_elapsed_steps[0])
            self._true_elapsed_steps[0] = 0
        
        # 调用父类 flush_trajectory（标准逻辑）
        super().flush_trajectory(verbose, ignore_empty_transition, env_idxs_to_flush)
    
    def close(self) -> None:
        """重写 close 来更新 JSON 中的所有 episodes 的真实执行步数"""
        # 先调用父类 close 保存数据
        super().close()
        
        # 如果使用了记录频率控制，更新 JSON 元数据
        if self.save_trajectory and self.record_freq is not None and self.record_freq < self.control_freq:
            try:
                import json
                from mani_skill.utils.io_utils import dump_json
                
                # 读取 JSON
                with open(self._json_path, 'r') as f:
                    json_data = json.load(f)
                
                # 为所有 episodes 添加真实执行步数
                for i, episode in enumerate(json_data['episodes']):
                    if i < len(self._episode_true_steps):
                        true_steps = self._episode_true_steps[i]
                        recorded_steps = episode['elapsed_steps']
                        
                        episode['true_elapsed_steps'] = true_steps
                        episode['recorded_steps'] = recorded_steps
                        episode['elapsed_steps'] = true_steps  # 更新为真实步数
                        episode['recording_frequency'] = f"{self.record_freq}Hz"
                        episode['sampling_interval'] = int(self.control_freq / self.record_freq)
                
                # 保存更新后的 JSON
                dump_json(self._json_path, json_data, indent=2)
                print(f"\n✅ updated metadata for {len(json_data['episodes'])} episodes")
            except Exception as e:
                print(f"⚠️  failed to update JSON metadata: {e}")

def create_gello_agent(port: str) -> GelloAgent:
    """创建并配置 GELLO agent"""
    
    # Panda 的 GELLO 配置
    config = DynamixelRobotConfig(
        joint_ids=[1, 2, 3, 4, 5, 6, 7],
        joint_offsets=[1.571, 3.142, 0.000, 4.712, 3.142, 4.712, 3.142],
        joint_signs=[1, 1, 1, 1, 1, -1, 1],
        gripper_config=[8, 159.62109375, 201.42109375]
    )
    
    gello = GelloAgent(port=port, dynamixel_config=config)
    print(f"✓ GELLO 已连接到 {port}")
    return gello

def wait_for_gello_alignment(env: BaseEnv, gello: GelloAgent, position_threshold: float = 0.2) -> bool:
    """
    等待用户将 GELLO 移动到接近机器人初始位置
    使用 TransformWindow 的 ghost 功能显示目标位置
    
    Returns:
        True: 开始记录
        False: 跳过此 episode
    """
    import time
    import sapien.utils.viewer
    from mani_skill.utils import sapien_utils
    
    viewer = env.render_human()
    robot = env.unwrapped.agent.robot
    
    # 获取初始位置
    # after reset, the robot is in the initial position
    initial_qpos = robot.get_qpos()
    if hasattr(initial_qpos, 'cpu'):
        initial_qpos = initial_qpos.cpu().numpy()
    else:
        initial_qpos = np.array(initial_qpos)
    target_joints = np.asarray(initial_qpos).flatten()[:7]
    
    # 获取 TransformWindow 并创建 ghost
    transform_window = None
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
            break
    
    if transform_window:
        # 选中机器人的 hand link（和 interactive_panda.py 一样）

        # link_names = [link.name for link in env.agent.robot.links]
        # print(link_names)

        # # 或者更简洁地打印
        # for name in link_names:
        #     print(name)

        
        # import pdb; pdb.set_trace()
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "robotiq_arg2f_base_link")._objs[0].entity)
        # 启用 TransformWindow 并创建 ghost
        transform_window.enabled = True
        transform_window.update_ghost_objects()
        # 关键：设置 follow=False，这样 ghost 固定在初始位置，不会跟随真实机器人！
        transform_window.follow = False
        # 渲染一次确保 ghost 显示
        env.render_human()
    
    try:
        while True:
            # 读取GELLO当前状态
            gello_joints = gello.act({})
            current_joints = gello_joints[:7]
            
            # 更新真实机器人到GELLO当前姿态
            qpos = robot.get_qpos()
            if hasattr(qpos, 'cpu'):
                new_qpos = qpos.cpu().numpy()
            else:
                new_qpos = np.array(qpos)
            new_qpos = new_qpos.flatten()  # 确保是一维数组
            new_qpos[:7] = current_joints
            new_qpos[-2:] = 0.04
            robot.set_qpos(new_qpos)
            
            # 计算误差
            joint_errors = np.abs(current_joints - target_joints)
            max_error = np.max(joint_errors)
            aligned = max_error < position_threshold
            
            env.render_human()
            
            time.sleep(0.05)

            if aligned:
                return True
    
    finally:
        # 恢复初始姿态，清理ghost
        robot.set_qpos(initial_qpos)
        if transform_window:
            transform_window.enabled = False
            viewer.select_entity(None)  # 取消选中
    
    return True

def gello_to_maniskill_action(gello_joints: np.ndarray, env: BaseEnv, debug: bool = False) -> np.ndarray:
    """
    convert GELLO joint states to ManiSkill action
    
    Args:
        gello_joints: [8] - GELLO 关节状态 [θ₁...θ₇, gripper_0to1]
        env: ManiSkill environment
        debug: whether to print debug information
    
    Returns:
        action: ManiSkill action, format depends on the action space of the environment
    """
    # check the action space of the environment
    action_dim = env.action_space.shape[0]
    
    if debug:
        print(f"\n[DEBUG] GELLO joints: {gello_joints}")
        print(f"[DEBUG] Action dim: {action_dim}")
        print(f"[DEBUG] Action space: {env.action_space}")
    
    if action_dim == 8:
        # Panda has 7 joints + 1 gripper
        action = gello_joints.copy()
        
        # Get gripper controller to check its limits
        controller = env.base_env.agent.controller
        gripper_controller = None
        
        # Find gripper controller (could be in a dict for combined controllers)
        if hasattr(controller, 'controllers') and isinstance(controller.controllers, dict):
            if 'gripper' in controller.controllers:
                gripper_controller = controller.controllers['gripper']
        elif hasattr(controller, 'config') and hasattr(controller.config, 'lower'):
            # Single controller case (unlikely for panda, but handle it)
            gripper_controller = controller
        
        if gripper_controller is not None:
            # Get gripper limits from action space (already accounts for normalization)
            action_space = gripper_controller.single_action_space
            gripper_low = action_space.low[0]
            gripper_high = action_space.high[0]
            
            # Check if action space is normalized ([-1, 1]) or actual range
            if gripper_low >= -1.01 and gripper_high <= 1.01:
                # Action space is normalized, map 0-1 to -1 to 1
                # GELLO: 0 (open) -> 1 (closed)
                # ManiSkill normalized: -1 (open) -> 1 (closed)
                action[7] = 2.0 * gello_joints[7] - 1.0
                action[7] = -action[7]
                action[7] = min(action[7], 0.5)
          
            else:
                # Action space is actual range, map 0-1 to actual range
                # GELLO: 0 (open) -> 1 (closed)
                # ManiSkill: gripper_low (open) -> gripper_high (closed)
                action[7] = gripper_low + (gripper_high - gripper_low) * gello_joints[7]
            
            if debug:
                print(f"[DEBUG] Gripper controller found: {type(gripper_controller).__name__}")
                print(f"[DEBUG] Gripper action space: [{gripper_low}, {gripper_high}]")
                print(f"[DEBUG] GELLO gripper: {gello_joints[7]} -> ManiSkill gripper: {action[7]}")
        else:
            # Fallback: assume normalized action space
            if debug:
                print(f"[DEBUG] Gripper controller not found, using default mapping")
            action[7] = 2.0 * gello_joints[7] - 1.0
        
        if debug:
            print(f"[DEBUG] Final action: {action}")
        
    else:
        raise ValueError(f"unsupported action dimension: {action_dim}")
    
    return action

def solve_with_gello(env: BaseEnv, gello: GelloAgent, control_freq: float = 30.0):
    """
    使用 GELLO 进行 teleoperation
    
    Args:
        env: ManiSkill environment
        gello: GELLO agent
        control_freq: control frequency (Hz)
    
    Returns:
        code: "quit", "continue", or "restart"
    """
    import time
    
    # ensure the environment uses the correct control mode
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], f"unsupported control mode: {env.unwrapped.control_mode}"
    
    # render the environment
    viewer = env.render_human()
    
    step = 0
    dt = 1.0 / control_freq
    
    try:
        while True:
            step_start = time.time()
            
            # ==========================================
            # 1. check keyboard input
            # ==========================================
            if viewer.window.key_press("h"):
                print("""
                    h: show this help menu
                    q: quit the script (won't save current episode)
                    r: restart the current episode (without saving)
                    
                    Note: Episodes are ONLY saved when task success is achieved!
                    
                    move GELLO to control the robot's joints and gripper!
                """)
                continue
            elif viewer.window.key_press("q"):
                print("\nquitting...")
                return "quit"
            elif viewer.window.key_press("r"):
                print("\nrestarting episode")
                return "restart"
            
            # ==========================================
            # 2. read the joint states from GELLO
            # ==========================================
            gello_joints: Any = gello.act({})
            # gello_joints: [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆, θ₇, gripper_0to1]
            
            # ==========================================
            # 3. convert to ManiSkill action
            # ==========================================
            # debug = (step % 30 == 0) 
            action: ndarray = gello_to_maniskill_action(gello_joints, env, debug=False)
            
            # ==========================================
            # 4. execute the action
            # ==========================================
            obs, reward, terminated, truncated, info = env.step(action)
            
            # ==========================================
            # 5. render the environment
            # ==========================================
            env.render_human()

            # ==========================================
            # 6. if successful, save the trajectory
            # ==========================================
            if info.get('success', False):
                return "continue"

            # ==========================================
            # 7. check if the episode is terminated
            # ==========================================
            if terminated or truncated:
                print(f"\n❌ Episode terminated but not successful (step: {step})")
                print("press 'r' to restart, or 'q' to quit (won't save)")
                # 只允许 success 才保存，失败的 episode 不能手动保存
                while True:
                    env.render_human()
                    if viewer.window.key_press("r"):
                        return "restart"
                    elif viewer.window.key_press("q"):
                        return "quit"
                    import time
                    time.sleep(0.01)
            
            # ==========================================
            # 7. control frequency
            # ==========================================
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            step += 1
            
    except KeyboardInterrupt:
        print(f"\n\nKeyboardInterrupt - episode interrupted at step {step}")
        return "restart"

def main(args: Args):
    """main function"""
    
    # ==========================================
    # 1. create GELLO agent
    # ==========================================
    print("initializing GELLO...")
    gello = create_gello_agent(args.gello_port)
    
    # ==========================================
    # 2. create ManiSkill environment
    # ==========================================
    print(f"creating environment: {args.env_id}")
    
    # 如果使用时间戳，添加到路径中
    if args.use_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.record_dir}/{args.env_id}/gello_teleop_{timestamp}/"
        print(f"✓ 使用时间戳: {timestamp}")
    else:
        output_dir = f"{args.record_dir}/{args.env_id}/gello_teleop/"
    
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",  # important: use joint position control
        render_mode="rgb_array",
        reward_mode="none",
        robot_uids=args.robot_uid,
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
        sensor_configs=dict(
            hand_camera=dict(width=256, height=256)  # 覆盖手部相机分辨率为256x256
        ),
        max_episode_steps=args.max_episode_steps  # configurable maximum steps
    )
    
    # 使用 RecordEpisodeWithFreq 支持频率控制
    print("✓ using joint_pos save mode with frequency control")
    print(f"  action format: [7 joints + 1 gripper]")
    if args.record_freq is not None:
        print(f"  recording frequency: {args.record_freq}Hz (control frequency: {args.control_freq}Hz)")
    else:
        print(f"  recording frequency: every step (control frequency: {args.control_freq}Hz)")
    env = RecordEpisodeWithFreq(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=args.save_video,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via GELLO hardware (joint_pos)",
        record_freq=args.record_freq,
        control_freq=args.control_freq
    )
    
    # ==========================================
    # 3. data collection loop
    # ==========================================
    tbar = tqdm(total=args.target_trajs, desc="Collecting trajectories", leave=True)
    num_trajs = 0
    seed = torch.randint(0, 1000000, (1,)).item()
    env.reset(seed=seed)  # 初始 reset，不保存（因为 save_on_reset=False）
    
    print(f"\nstarting data collection...")
    print(f"data will be saved to: {output_dir}")
    
    while True:
        # wait for GELLO to align to the initial position
        aligned = wait_for_gello_alignment(env, gello, position_threshold=0.2)
        if not aligned:
            # user chose to skip, reset the environment (不保存)
            seed = torch.randint(0, 1000, (1,)).item()
            # 清空 buffer（如果存在），然后 reset
            if env._trajectory_buffer is not None:
                env._trajectory_buffer = None
            env.reset(seed=seed)
            continue
        
        code = solve_with_gello(env, gello, control_freq=args.control_freq)
        
        if code == "quit":
            # 清空 trajectory buffer（不保存未完成的 episode）
            if env._trajectory_buffer is not None:
                env._trajectory_buffer = None
            break
        elif code == "continue":
            # Success! 手动保存轨迹和视频，然后 reset
            print("✅ 任务成功！保存轨迹和视频...")
            
            # 保存视频
            if env.save_video:
                env.flush_video()
                print(f"  ✓ 视频已保存")
            
            # 保存轨迹
            if env._trajectory_buffer is not None:
                env.flush_trajectory()
                print(f"  ✓ 轨迹已保存")
            
            seed = torch.randint(0, 1000, (1,)).item()
            num_trajs += 1
            env.reset(seed=seed)
            tbar.update(1)
            continue
        elif code == "restart":
            # 清空 buffer，不保存失败的 episode
            if env._trajectory_buffer is not None:
                env._trajectory_buffer = None
            env.reset(seed=seed)
            continue  # 重要：继续循环

    # ==========================================
    # 4. save data
    # ==========================================
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    
    print(f"\n{'='*60}")
    print(f"data collection completed!")
    print(f"{'='*60}")
    print(f"number of trajectories: {num_trajs}")
    print(f"data saved to: {h5_file_path}")
    print(f"metadata saved to: {json_file_path}")
    
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
