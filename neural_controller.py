#!/usr/bin/env python3
"""
Neural network controller using GRPO (Group Relative Policy Optimization)
to train a policy for grabbing a cube and dropping it into a box.
"""

import random

import mujoco
import numpy as np
import torch
import torch.nn as nn

from utils import FRAME_RATE, HEIGHT, WIDTH, load_model, save_video

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training hyperparameters
num_updates = 100
num_groups = 8
num_trajectories = 8
save_every = 20
initial_entropy_coef = 0.01
entropy_decay = 0.995

# Box dimensions (from scene.xml)
BOX_POS = np.array([0.25, -0.2, 0.0])  # Box center position
BOX_HEIGHT = 0.0275  # Top of box (0.005 bottom + 0.025 height + 0.0025 for geoms)
BOX_SIZE = 0.08  # Half-size of box


class ObservationNormalizer:
    """Normalizes observations using exponential moving average."""

    def __init__(self, obs_dim, alpha=0.99):
        self.alpha = alpha
        self.running_mean = np.zeros(obs_dim, dtype=np.float32)
        self.running_var = np.ones(obs_dim, dtype=np.float32)
        self.count = 0

    def update(self, obs):
        """Update running statistics."""
        self.count += 1
        if self.count == 1:
            self.running_mean = obs.copy()
            self.running_var = np.ones_like(obs)
        else:
            delta = obs - self.running_mean
            self.running_mean += (1 - self.alpha) * delta
            self.running_var = (
                self.alpha * self.running_var + (1 - self.alpha) * delta**2
            )

    def normalize(self, obs):
        """Normalize observation."""
        std = np.sqrt(self.running_var) + 1e-8
        return (obs - self.running_mean) / std


class PolicyNetwork(nn.Module):
    """Gaussian policy network with LayerNorm."""

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()

        layers = []
        input_dim = obs_dim

        # Build hidden layers with LayerNorm
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Output heads for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

    def forward(self, obs):
        """Forward pass."""
        features = self.feature_extractor(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        # Clamp log_std to [-2, 1]
        log_std = torch.clamp(log_std, -2.0, 1.0)
        return mean, log_std

    def sample(self, obs):
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs, action):
        """Evaluate log probability of action."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


def get_observations(model, data, device):
    """
    Extract observations from MuJoCo simulation.
    Returns: normalized observation vector
    """
    # Get body IDs
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_box")
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    # Extract positions
    cube_pos = data.xpos[cube_body_id].copy()
    box_pos = data.xpos[box_body_id].copy()
    gripper_pos = data.xpos[gripper_body_id].copy()

    # Get joint positions and velocities (6 actuators)
    joint_pos = data.qpos[:6].copy() if len(data.qpos) >= 6 else np.zeros(6)
    joint_vel = data.qvel[:6].copy() if len(data.qvel) >= 6 else np.zeros(6)

    # Concatenate all observations
    obs = np.concatenate(
        [
            joint_pos,  # 6
            joint_vel,  # 6
            cube_pos,  # 3
            box_pos,  # 3
            gripper_pos,  # 3
            cube_pos - box_pos,  # 3 (relative position)
            gripper_pos - cube_pos,  # 3 (gripper to cube)
        ]
    )

    return obs.astype(np.float32)


def compute_reward(model, data, trajectory):
    """
    Compute reward for a trajectory.
    Priority order (most to least important):
    1. Cube is on top of box = success
    2. Cube is getting closer to box
    3. Cube is at height at least greater than height of box while far away from box
    4. Cube is at height closer to ground while closer to box
    5. Distance between gripper + cube
    """
    total_reward = 0.0

    for step_data in trajectory:
        # Get current state
        cube_pos = step_data["cube_pos"]
        gripper_pos = step_data["gripper_pos"]

        # 1. Cube on box = success (highest priority)
        cube_to_box_xy = np.linalg.norm(cube_pos[:2] - BOX_POS[:2])
        cube_z = cube_pos[2]
        box_top_z = BOX_POS[2] + BOX_HEIGHT

        if (
            cube_to_box_xy < BOX_SIZE
            and cube_z > box_top_z - 0.01
            and cube_z < box_top_z + 0.05
        ):
            total_reward += 100.0  # Success reward

        # 2. Cube getting closer to box (distance reward)
        cube_to_box_dist = np.linalg.norm(cube_pos - BOX_POS)
        total_reward += -0.1 * cube_to_box_dist  # Negative distance (closer = better)

        # 3. Height management: higher when far, lower when close
        cube_to_box_xy_dist = np.linalg.norm(cube_pos[:2] - BOX_POS[:2])
        if cube_to_box_xy_dist > 0.15:  # Far from box
            # Reward being higher than box height
            if cube_z > box_top_z:
                total_reward += 2.0 * (cube_z - box_top_z)
        else:  # Close to box
            # Reward being closer to ground/box height
            height_reward = -1.0 * abs(cube_z - box_top_z)
            total_reward += height_reward

        # 4. Gripper-cube distance (penalty for large distances)
        gripper_to_cube_dist = np.linalg.norm(gripper_pos - cube_pos)
        total_reward += -0.5 * gripper_to_cube_dist

    return total_reward


def collect_trajectories(
    policy, num_trajectories, episode_length, normalizer, model, data
):
    """
    Collect trajectories for training.
    Returns: list of trajectories, each containing observations, actions, rewards, dones
    """
    trajectories = []

    for traj_idx in range(num_trajectories):
        # Random episode length between 5-10 seconds
        if episode_length is None:
            episode_duration = random.uniform(5.0, 10.0)  # seconds
        else:
            episode_duration = episode_length  # if provided, assume it's in seconds

        # Reset environment
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        # Track episode start time
        start_time = data.time
        target_time = start_time + episode_duration

        trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "cube_pos": [],
            "gripper_pos": [],
        }

        # Run episode until target time is reached
        while data.time < target_time:
            # Get observation
            obs = get_observations(model, data, device)

            # Normalize observation
            normalizer.update(obs)
            obs_normalized = normalizer.normalize(obs)

            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(device)

            # Sample action from policy
            with torch.no_grad():
                action, log_prob = policy.sample(obs_tensor)
                action_np = action.cpu().numpy()[0]

            # Apply action (clamp to reasonable ranges)
            action_clamped = np.clip(action_np, -1.0, 1.0)
            data.ctrl[:6] = action_clamped

            # Store positions for reward computation
            cube_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
            )
            gripper_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
            )
            cube_pos = data.xpos[cube_body_id].copy()
            gripper_pos = data.xpos[gripper_body_id].copy()

            # Step simulation
            mujoco.mj_step(model, data)

            # Store data
            trajectory["observations"].append(obs_normalized)
            trajectory["actions"].append(action_np)
            trajectory["cube_pos"].append(cube_pos)
            trajectory["gripper_pos"].append(gripper_pos)
            trajectory["dones"].append(False)

        # Compute trajectory reward
        traj_reward = compute_reward(
            model,
            data,
            [
                {"cube_pos": cp, "gripper_pos": gp}
                for cp, gp in zip(trajectory["cube_pos"], trajectory["gripper_pos"])
            ],
        )

        # Assign reward to all steps (sparse reward)
        num_steps = len(trajectory["observations"])
        trajectory["rewards"] = [0.0] * (num_steps - 1) + [traj_reward]

        trajectories.append(trajectory)

    return trajectories


def check_success(model, data, trajectory):
    """Check if task was successful (cube on box)."""
    # Check final cube position
    final_cube_pos = trajectory["cube_pos"][-1]
    cube_to_box_xy = np.linalg.norm(final_cube_pos[:2] - BOX_POS[:2])
    cube_z = final_cube_pos[2]
    box_top_z = BOX_POS[2] + BOX_HEIGHT

    return (
        cube_to_box_xy < BOX_SIZE
        and cube_z > box_top_z - 0.01
        and cube_z < box_top_z + 0.05
    )


def check_gripper_touch(model, data, trajectory):
    """Check if gripper touched cube."""
    for step_data in zip(trajectory["cube_pos"], trajectory["gripper_pos"]):
        cube_pos, gripper_pos = step_data
        dist = np.linalg.norm(gripper_pos - cube_pos)
        if dist < 0.05:  # Threshold for touching
            return True
    return False


def main():
    """Main training loop."""
    # Load model
    model, data = load_model()

    # Get observation dimension
    test_obs = get_observations(model, data, device)
    obs_dim = len(test_obs)
    action_dim = 6  # 6 actuators

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize policy and normalizer
    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    normalizer = ObservationNormalizer(obs_dim)

    # Entropy coefficient (decays over time)
    entropy_coef = initial_entropy_coef

    # EMA for average reward tracking
    ema_reward = 0.0
    ema_alpha = 0.99

    # Statistics tracking
    success_count = 0
    touch_count = 0
    total_episodes = 0

    # Training loop
    for update in range(num_updates):
        print(f"\n=== Update {update + 1}/{num_updates} ===")

        all_advantages = []
        all_log_probs = []
        all_entropies = []
        update_successes = 0
        update_touches = 0
        update_rewards = []

        # Collect trajectories in groups
        for group_idx in range(num_groups):
            # Collect trajectories for this group
            trajectories = collect_trajectories(
                policy, num_trajectories, None, normalizer, model, data
            )

            # Compute trajectory rewards
            traj_rewards = [np.sum(traj["rewards"]) for traj in trajectories]
            group_mean_reward = np.mean(traj_rewards)

            # Compute advantages (trajectory_reward - group_mean_reward)
            advantages = [r - group_mean_reward for r in traj_rewards]

            # Normalize advantages
            if len(advantages) > 1:
                adv_mean = np.mean(advantages)
                adv_std = np.std(advantages) + 1e-8
                advantages = [(a - adv_mean) / adv_std for a in advantages]

            # Track statistics
            for traj in trajectories:
                total_episodes += 1
                if check_success(model, data, traj):
                    update_successes += 1
                    success_count += 1
                if check_gripper_touch(model, data, traj):
                    update_touches += 1
                    touch_count += 1
                update_rewards.append(np.sum(traj["rewards"]))

            # Prepare for policy update (only positive advantages)
            for traj, adv in zip(trajectories, advantages):
                if adv > 0:  # Only update on positive advantages
                    obs_tensor = torch.FloatTensor(np.array(traj["observations"])).to(
                        device
                    )
                    action_tensor = torch.FloatTensor(np.array(traj["actions"])).to(
                        device
                    )

                    log_probs, entropies = policy.evaluate(obs_tensor, action_tensor)

                    # Expand advantage to match number of steps in trajectory
                    traj_length = len(log_probs)
                    expanded_adv = torch.full(
                        (traj_length,), adv, dtype=torch.float32
                    ).to(device)

                    all_log_probs.append(log_probs)
                    all_entropies.append(entropies.mean())
                    all_advantages.append(expanded_adv)

        # Policy update
        if len(all_log_probs) > 0:
            # Concatenate all data
            log_probs_tensor = torch.cat(all_log_probs)
            advantages_tensor = torch.cat(all_advantages)
            entropies_tensor = torch.stack(all_entropies)

            # Compute loss
            policy_loss = -(log_probs_tensor * advantages_tensor).mean()
            entropy_bonus = entropy_coef * entropies_tensor.mean()
            total_loss = policy_loss - entropy_bonus

            # Update policy
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # Decay entropy coefficient
            entropy_coef *= entropy_decay

        # Update EMA reward
        if len(update_rewards) > 0:
            mean_reward = np.mean(update_rewards)
            if ema_reward == 0.0:
                ema_reward = mean_reward
            else:
                ema_reward = ema_alpha * ema_reward + (1 - ema_alpha) * mean_reward

        # Print statistics
        success_rate = update_successes / (num_groups * num_trajectories) * 100
        touch_rate = update_touches / (num_groups * num_trajectories) * 100

        print(f"Mean reward: {np.mean(update_rewards):.2f}")
        print(f"EMA reward: {ema_reward:.2f}")
        print(f"Touch rate: {touch_rate:.1f}%")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Entropy coef: {entropy_coef:.6f}")

        # Save video every save_every updates
        if (update + 1) % save_every == 0:
            print(f"\nSaving video at update {update + 1}...")
            # Collect one trajectory for video (10 seconds)
            trajectories = collect_trajectories(
                policy, 1, 10.0, normalizer, model, data
            )

            # Render video
            model_vid, data_vid = load_model()
            frames = []

            # Replay trajectory
            mujoco.mj_resetData(model_vid, data_vid)
            mujoco.mj_forward(model_vid, data_vid)

            traj = trajectories[0]
            start_time = data_vid.time
            with mujoco.Renderer(model_vid, height=HEIGHT, width=WIDTH) as renderer:
                for step, obs_norm in enumerate(traj["observations"]):
                    obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _ = policy.sample(obs_tensor)
                        action_np = action.cpu().numpy()[0]

                    data_vid.ctrl[:6] = np.clip(action_np, -1.0, 1.0)

                    # Step simulation
                    mujoco.mj_step(model_vid, data_vid)

                    # Capture frame at desired framerate
                    current_time = data_vid.time - start_time
                    if len(frames) < current_time * FRAME_RATE:
                        renderer.update_scene(data_vid, camera="closeup")
                        pixels = renderer.render()
                        frames.append(pixels)

            # Save video
            video_name = f"neural_{update + 1:04d}.mp4"
            save_video(frames, video_name)
            print(f"Video saved: {video_name}")

    print("\n=== Training Complete ===")
    final_success_rate = (
        success_count / total_episodes * 100 if total_episodes > 0 else 0
    )
    final_touch_rate = touch_count / total_episodes * 100 if total_episodes > 0 else 0
    print(f"Final success rate: {final_success_rate:.1f}%")
    print(f"Final touch rate: {final_touch_rate:.1f}%")


if __name__ == "__main__":
    main()
