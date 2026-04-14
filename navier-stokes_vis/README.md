# Forward and Inverse Modeling of the 3D Incompressible (Steady-State) Navier–Stokes Equations with Neural Networks ... A Story in Machine Learning-Driven Computational Fluid Dynamics

## Requirements

- Python 3.10+
- Node.js 18+
- Git LFS

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/mmarcotullio/navier-stokes_vis.git
cd navier-stokes_vis
```

**2. Pull model files**
```bash
git lfs install
git lfs pull
```

**3. Create Python virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Install Node dependencies**
```bash
cd main_vis/app && npm install
cd ../../story/app && npm install
cd ../..
```

**5. Run**
```bash
./start.sh
```
