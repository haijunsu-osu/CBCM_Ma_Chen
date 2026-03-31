# CBCM Four-Bar Mechanism Simulation

This project implements a high-fidelity kinetostatic simulation of a partially compliant four-bar mechanism using the **Chained Beam-Constraint Model (CBCM)**.

The simulation is based on **Example 4.2.1** from the reference paper:
> *Ma, F., & Chen, G. (2016). A Chained Beam-Constraint Model for Compliant Mechanisms. Journal of Mechanisms and Robotics.*

## Features

- **CBCM Solver**: A robust Python implementation with adaptive sub-stepping to ensure 100% convergence across the full cycle.
- **Interactive Visualization**: A premium HTML5 Canvas animation with real-time torque plotting and playback controls.
- **Physics Matching**: Parameters (material, geometry, and thickness) refined to match the 0.5 Nm torque peak and bistable behavior reported in the literature.

## Live Demo

If this repository is hosted on GitHub Pages, you can run the interactive simulation directly in your browser:
**[https://haijunsu-osu.github.io/CBCM_Ma_Chen/index.html](https://haijunsu-osu.github.io/CBCM_Ma_Chen/index.html)**

## How to Run Locally

### Prerequisites
- **Python 3.x**
- **NumPy, SciPy**

Install dependencies:
```bash
pip install numpy scipy
```

### Step 1: Generate Simulation Data
Run the Python script to compute the mechanism's motion. This generates `mechanism_data.json`.
```bash
python cbcm_fourbar.py
```

### Step 2: Start a Local Server
Host the files using Python's built-in server:
```bash
python -m http.server 8000
```

### Step 3: View the Animation
Open your browser to: [http://localhost:8000/index.html](http://localhost:8000/index.html)

## Mechanism Parameters (Reference Example 4.2.1)

- **Beam length ($L$):** 100 mm (Polypropylene, $E = 1.4$ GPa, $t = 1.7$ mm)
- **Crank length ($L_{AB}$):** ~29.3 mm
- **Ground spacing ($L_{DA}$):** ~75.7 mm
- **Coupler shape:** Rigid assembly $B-C-Q$ where $L_{BC}=L$, $L_{CQ}=L/20$, and $\theta_2 = 135^\circ$.
- **Boundary Condition:** Point $D$ is a clamped base (fixed vertical support).

## Citation

If you use this code or model in your research, please cite the original paper:

**Ma, F., & Chen, G. (2016). A Chained Beam-Constraint Model for Compliant Mechanisms. *Journal of Mechanisms and Robotics*, 8(2), 021018. doi:10.1115/1.4031641**
