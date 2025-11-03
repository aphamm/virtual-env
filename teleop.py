#!/usr/bin/env python3
"""
Render a 6 second video of the MuJoCo scene with SO-101 arm, red cube, and blue box.
"""

import mujoco
import numpy as np

from utils import DURATION, FRAME_RATE, HEIGHT, WIDTH, load_model, save_video


def smoothstep(t):
    """
    Smoothstep function: smooth interpolation with ease-in/ease-out.
    Input t should be in [0, 1].
    Returns value in [0, 1] with smooth transitions at boundaries.

    Formula: 3t² - 2t³
    - Starts and ends with zero velocity
    - Provides smooth acceleration and deceleration
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def interpolate_smoothstep(time, waypoints, waypoint_times):
    """
    Smoothly interpolate between waypoints using smoothstep function.
    Provides smooth acceleration at the start and deceleration at the end of each segment.

    Args:
        time: Current time
        waypoints: Array of waypoint values (shape: [n_waypoints, n_controls])
        waypoint_times: Array of times for each waypoint (shape: [n_waypoints])

    Returns:
        Interpolated control values with smooth transitions
    """
    if time <= waypoint_times[0]:
        return waypoints[0]
    if time >= waypoint_times[-1]:
        return waypoints[-1]

    # Find the segment containing the current time
    right = 0
    while right < len(waypoint_times) - 1 and waypoint_times[right] < time:
        right += 1
    left = right - 1

    # Calculate normalized time in [0, 1] for this segment
    t = (time - waypoint_times[left]) / (waypoint_times[right] - waypoint_times[left])

    # Apply smoothstep to get smooth interpolation parameter
    alpha = smoothstep(t)

    # Interpolate between waypoints
    return waypoints[left] * (1 - alpha) + waypoints[right] * alpha


def robot_control(model, data, time):
    """
    Control function for the robot arm with smooth interpolation between waypoints.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        time: Current simulation time

    Returns:
        numpy array of control signals for actuators

    Actuator order (from so101_new_calib.xml):
        0: shoulder_pan   - Base rotation (range: -1.92 to 1.92 rad)
        1: shoulder_lift  - Shoulder up/down (range: -1.75 to 1.75 rad)
        2: elbow_flex     - Elbow bend (range: -1.69 to 1.69 rad)
        3: wrist_flex     - Wrist bend (range: -1.66 to 1.66 rad)
        4: wrist_roll     - Wrist rotation (range: -2.74 to 2.84 rad)
        5: gripper        - Gripper open/close (range: -0.17 to 1.75 rad, 0=closed, 1.75=open)
    """

    waypoint_times = np.array([0.0, 1.0, 2.0, 2.3, 2.5, 3.0, 3.2, 4.0, 5.5])

    waypoints = np.array(
        [
            # approach slowly
            [0.03, 0.025, 0.025, 0.15, 0.15, 0.15],
            # intermediate position
            [0.08, 0.15, 0.15, 0.5, 1.0, 0.4],
            # hover over cube
            [0.09, 0.3, 0.25, 1.1, 1.8, 0.7],
            # grip cube
            [0.0, 0.3, 0.25, 1.1, 1.8, 0.0],
            # hold cube
            [0.0, 0.3, 0.25, 1.1, 1.8, 0.0],
            # lift up
            [0.09, -0.5, 0.25, 1.1, 1.8, 0.0],
            # hold cube
            [0.09, -0.5, 0.25, 1.1, 1.8, 0.0],
            # rotate over box
            [0.8, -0.5, 0.25, 1.1, 1.8, 0.0],
            # open gripper
            [0.8, -0.5, 0.25, 1.1, 1.8, 0.7],
        ]
    )

    return interpolate_smoothstep(time, waypoints, waypoint_times)


def main(duration=DURATION):
    """
    Render a video of the MuJoCo scene simulation with robot control.

    Args:
        duration: Duration of simulation in seconds
    """

    model, data = load_model()

    # Simulate and capture frames
    frames = []
    print(f"Actuators: {[model.actuator(i).name for i in range(model.nu)]}")

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        while data.time < duration:
            ctrl = robot_control(model, data, data.time)
            data.ctrl[:] = ctrl

            # Step the simulation
            mujoco.mj_step(model, data)

            # Capture frame at desired framerate
            if len(frames) < data.time * FRAME_RATE:
                renderer.update_scene(data, camera="closeup")
                pixels = renderer.render()
                frames.append(pixels)

    print(f"Rendered {len(frames)} frames over {duration} seconds")
    save_video(frames, "teleop.mp4")

    return


if __name__ == "__main__":
    main()
