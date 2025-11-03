#!/usr/bin/env python3
"""
Interactive MuJoCo scene viewer with robot arm control.
Allows you to interactively explore the scene and control the robot arm.

The viewer opens an interactive window where you can:
- Rotate camera: Left click + drag
- Pan camera: Right click + drag
- Zoom: Scroll wheel
- Control robot: Edit control values in code below or use the interactive viewer controls
"""

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Path to scene XML file (relative to project root)
SCENE_PATH = PROJECT_ROOT / "env" / "scene.xml"

def interactive_viewer(enable_control=True):
    """
    Launch an interactive MuJoCo viewer for the scene.
    
    Args:
        enable_control: If True, applies control from get_robot_control()
                       If False, robot remains in default position
    """
    # Load model
    if not SCENE_PATH.exists():
        print(f"Error: Scene file not found at {SCENE_PATH}")
        print("Please ensure the env/scene.xml file exists.")
        return
    
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    
    # Open interactive viewer
    with mujoco.viewer.launch(model, data) as viewer:
        print("Viewer opened! Start exploring...\n")
        
        while viewer.is_running():
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer (updates display)
            viewer.sync()
            
            # Small delay to prevent overwhelming the system
            time.sleep(model.opt.timestep)
    
    print("\nViewer closed. Goodbye!")


if __name__ == "__main__":
    try:
        # Set enable_control=False to just view the scene without robot control
        interactive_viewer(enable_control=True)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
