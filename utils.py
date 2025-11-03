from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco

PROJECT_ROOT = Path(__file__).parent.absolute()
SCENE_PATH = PROJECT_ROOT / "env" / "scene.xml"
OUTPUT_PATH = PROJECT_ROOT / "output"
FRAME_RATE = 30
HEIGHT = 1080
WIDTH = 1920
DURATION = 6.0


def load_model():
    # Load model
    if not SCENE_PATH.exists():
        print(f"Error: Scene file not found at {SCENE_PATH}")
        print("Please ensure the env/scene.xml file exists.")
        return

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # Reset state and time.
    mujoco.mj_resetData(model, data)

    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)

    return model, data


def save_video(frames, output_name):
    # Create animation - remove borders
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # Inverted for image coordinates
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins
    im = ax.imshow(frames[0], aspect="auto", interpolation="nearest")

    def animate(frame_num):
        im.set_array(frames[frame_num])
        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        interval=1000 / FRAME_RATE,
        blit=True,
        repeat=False,
    )

    # Save animation
    try:
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=FRAME_RATE, metadata=dict(artist="MuJoCo"), bitrate=1000)
        output_path = OUTPUT_PATH / output_name
        anim.save(str(output_path), writer=writer, savefig_kwargs={"pad_inches": 0})
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"FFmpeg writer not available ! {e}")

    plt.close()
