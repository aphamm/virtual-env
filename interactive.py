#!/usr/bin/env python3
"""
Interactive MuJoCo scene viewer with robot arm control.
Allows you to interactively explore the scene and control the robot arm.
"""

import time

import mujoco
import mujoco.viewer

from utils import load_model


def main():
    """
    Launch an interactive MuJoCo viewer for the scene.
    """

    model, data = load_model()

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


if __name__ == "__main__":
    main()
