#!/usr/bin/env python3
"""
Render a 5 second video of the MuJoCo scene with SO-101 arm, red cube, and blue box.
Based on tutorial.ipynb video rendering examples.
"""

import mujoco
import numpy as np
from pathlib import Path

# Try to import video writing libraries
try:
    import imageio
    USE_IMAGEIO = True
except ImportError:
    USE_IMAGEIO = False

# Fallback to matplotlib for saving
if not USE_IMAGEIO:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Path to scene XML file (relative to project root)
SCENE_PATH = PROJECT_ROOT / "env" / "scene.xml"

def render_video():
    """Render a 5 second video of the MuJoCo scene simulation."""
    # Load model from XML file
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    
    # Video parameters - higher resolution
    duration = 1.0   # (seconds)
    framerate = 60   # (Hz)
    height = 1080   # Full HD height
    width = 1920    # Full HD width
    
    # Simulate and capture frames
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while data.time < duration:
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
    
    # Save video to file
    output_path = PROJECT_ROOT / "scene_video.mp4"
    print(f"Saving video to: {output_path}")
    
    if USE_IMAGEIO:
        # Use imageio to save video (simplest method, no borders)
        # Use higher quality settings for better output
        imageio.mimwrite(str(output_path), frames, fps=framerate, 
                        codec='libx264', quality=10, pixelformat='yuv420p',
                        ffmpeg_params=['-crf', '18'])  # Lower CRF = higher quality
        print(f"Video saved successfully to: {output_path}")
    else:
        # Fallback to matplotlib animation
        print("Saving video using matplotlib...")
        
        # Create animation - remove borders
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
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
            output_path = PROJECT_ROOT / "scene_video.mp4"
        except (KeyError, AttributeError):
            # Fallback to pillow writer (saves as GIF)
            Writer = animation.writers['pillow']
            writer = Writer(fps=framerate)
            output_path = PROJECT_ROOT / "scene_video.gif"
        
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
        render_video()
    except Exception as e:
        print(f"Error rendering video: {e}")
        import traceback
        traceback.print_exc()