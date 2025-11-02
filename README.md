# 🤖 MuJoCo Robot Arm Simulation with RL

## 🎯 Goal
Create a MuJoCo simulation where a **LeRobot SO-101 Arm** grabs a **red cube** and drops it into a **blue box**, then train a neural network controller using reinforcement learning.

## 📋 Project Phases

### Phase 1: 🧪 Manual Control & Feasibility Testing
- [ ] Create MuJoCo scene with SO-101 arm, red cube, and blue box
- [ ] Ensure proper physics (collision detection, no interpenetration)
- [ ] Build manual control interface
- [ ] Test if task is achievable manually (grab cube → drop in box)

### Phase 2: 🧠 Basic Neural Network Controller
- [ ] Design input/output interface to MuJoCo simulator
- [ ] Create simple neural network (e.g., MLP) for control
- [ ] Implement reward function:
  - Distance to ball (grasping)
  - Ball height above box (transport)
  - Ball in box (success)
- [ ] Set up basic RL loop (policy gradient or value-based)
- [ ] Verify weight updates via backprop during training

### Phase 3: 🚀 Octo Integration
- [ ] Integrate [Octo: An Open-Source Generalist Robot Policy](https://github.com/octo-models/octo)
- [ ] Fine-tune Octo on task-specific simulation data
- [ ] Evaluate performance improvement vs. basic NN

## 🏗️ Project Structure
```
virtual-env/
├── env/              # MuJoCo environment & scene files
├── manual_control.py # Phase 1: Manual control interface
├── basic_rl/         # Phase 2: Basic neural network controller
├── octo_rl/          # Phase 3: Octo integration
└── refs/             # SO-101 arm reference files
```

## 🔧 Tech Stack
- **Physics**: MuJoCo
- **Robot**: LeRobot SO-101 Arm
- **RL Framework**: TBD (Stable-Baselines3 / JAX/Flax)
- **Policy Model**: Custom MLP → Octo

## 🔄 How to Reproduce

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd virtual-env
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Simulation

**Phase 1: Manual Control** (when implemented)
```bash
python manual_control.py
```

**Phase 2: Basic RL Training** (when implemented)
```bash
cd basic_rl
python train.py
```

**Phase 3: Octo Training** (when implemented)
```bash
cd octo_rl
python train_octo.py
```

---
**Status**: Starting Phase 1 🔄
