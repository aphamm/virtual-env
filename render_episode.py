#!/usr/bin/env python3
"""
Render a 5 second video of the MuJoCo scene with SO-101 arm, red cube, and blue box.
Based on tutorial.ipynb video rendering examples.
"""

import mujoco
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Path to scene XML file (relative to project root)
SCENE_PATH = PROJECT_ROOT / "env" / "scene.xml"

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
        numpy array of control signals for actuators [shoulder_pan, shoulder_lift, 
        elbow_flex, wrist_flex, wrist_roll, gripper]
    
    Actuator order (from so101_new_calib.xml):
        0: shoulder_pan   - Base rotation (range: -1.92 to 1.92 rad)
        1: shoulder_lift  - Shoulder up/down (range: -1.75 to 1.75 rad)
        2: elbow_flex     - Elbow bend (range: -1.69 to 1.69 rad)
        3: wrist_flex     - Wrist bend (range: -1.66 to 1.66 rad)
        4: wrist_roll     - Wrist rotation (range: -2.74 to 2.84 rad)
        5: gripper        - Gripper open/close (range: -0.17 to 1.75 rad, 0=closed, 1.75=open)
    """
    # Define waypoints at 0.5 second intervals
    # Each waypoint is [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    waypoint_times = np.array([0.0, 1.0, 2.0, 2.3, 2.5, 3.0, 3.2, 4.0, 5.5]) 
    
    waypoints = np.array([
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
    ])
    
    # Interpolate smoothly between waypoints using smoothstep
    ctrl = interpolate_smoothstep(time, waypoints, waypoint_times)
    return ctrl


def render_video(control_func=None, duration=6.0):
    """
    Render a video of the MuJoCo scene simulation with optional robot control.
    
    Args:
        control_func: Optional function(model, data, time) -> ctrl array
                     If None, robot will be uncontrolled
        duration: Duration of simulation in seconds
    """

    # Load model from XML file
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    
    # Video parameters - higher resolution
    framerate = 30   # (Hz)
    height = 1080   # Full HD height
    width = 1920    # Full HD width
    
    # Simulate and capture frames
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    print(f"Actuators: {[model.actuator(i).name for i in range(model.nu)]}")
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < duration:
            # Apply control if provided
            if control_func is not None:
                ctrl = control_func(model, data, data.time)
                data.ctrl[:] = ctrl
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Capture frame at desired framerate
            if len(frames) < data.time * framerate:
                # Use the closeup camera from scene.xml
                renderer.update_scene(data, camera="closeup")
                pixels = renderer.render()
                frames.append(pixels)
    
    print(f"Rendered {len(frames)} frames over {duration} seconds")
    print(f"Framerate: {framerate} Hz, Resolution: {width}x{height}")
    output_path = PROJECT_ROOT / "episode.mp4"
    print(f"Saving video to: {output_path}")
            
    # Create animation - remove borders
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=300)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Inverted for image coordinates
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins
    im = ax.imshow(frames[0], aspect='auto', interpolation='nearest')
    
    def animate(frame_num):
        im.set_array(frames[frame_num])
        return [im]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames),
        interval=1000/framerate, blit=True, repeat=False
    )
    
    # Save animation - try ffmpeg first, fallback to pillow
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=framerate, metadata=dict(artist='MuJoCo'), bitrate=8000)
        output_path = PROJECT_ROOT / "episode.mp4"
    except (KeyError, AttributeError):
        # Fallback to pillow writer (saves as GIF)
        Writer = animation.writers['pillow']
        writer = Writer(fps=framerate)
        output_path = PROJECT_ROOT / "episode.gif"
    
    # Save with no borders
    anim.save(str(output_path), writer=writer, 
                savefig_kwargs={'pad_inches': 0})
    print(f"Video saved to: {output_path}")
    plt.close()
    
    return frames

if __name__ == "__main__":
    
    # Check if scene file exists
    if not SCENE_PATH.exists():
        print(f"Error: Scene file not found at {SCENE_PATH}")
        print("Please ensure the env/scene.xml file exists.")
        exit(1)
    
    try:
        # Use the robot_control function to control the arm
        # Pass None to disable control, or modify robot_control() above to design your sequence
        render_video(control_func=robot_control)
    except Exception as e:
        print(f"Error rendering video: {e}")
        import traceback
        traceback.print_exc()