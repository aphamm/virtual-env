# 🤖 MuJoCo Robot Arm Simulation with RL

## 🎯 Goal
Create a MuJoCo simulation where a **LeRobot SO-101 Arm** grabs a **red cube** and drops it into a **blue box**, then train a neural network controller using reinforcement learning. 

## 🔄 How to Reproduce

### 0. Setup

Clone the repository

```bash
git clone git@github.com:aphamm/virtual-env.git
cd virtual-env
```

Install dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Phase 1. Running "Teleoperated" Episode

```bash
python teleop.py
python interactive.py # run interactive session
```

### Phase 2: Basic Neural Net + RL

```bash
python neural_controller.py
```

## 📋 Project Checklist

- [X] Create MuJoCo scene with SO-101 arm, red cube, and blue box
- [X] Test if task is achievable manually (grab cube → drop in box)
- [ ] Design input/output neural interface to simulator
- [ ] Design reward function with basic RL loop
- [ ] Integrate [Octo: An Open-Source Generalist Robot Policy](https://github.com/octo-models/octo)
- [ ] Fine-tune Octo on task-specific simulation data
