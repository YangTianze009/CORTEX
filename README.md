# CORTEX

A unified repository for reproducing interpretability experiments on **VQGAN** and **DALLE**.

---

## 1. Environment

| Requirement | Value      |
| ----------- | ---------- |
| Python      | **3.12.3** |
| Conda env   | **CORTEX** |

### 1.1 Create the environment

```bash
# Option A (preferred): use the YAML file
conda env create -f environment.yml   # creates env named “CORTEX”
conda activate CORTEX

# Option B: use the requirements file
conda create -n CORTEX python=3.12.3
conda activate CORTEX
pip install -r requirements.txt
```

## 2. Repository Layout
```text
CORTEX
├── VQGAN_explanation/   # Experiments & analyses based on VQGAN
├── Dalle_explanation/   # Experiments & analyses based on DALLE
├── environment.yml      # Conda environment specification (preferred)
├── requirements.txt     # Pip fallback dependency list
└── README.md            # Repository overview (you are here)
```

## 3. Experiments
### 3.1 VQGAN Experiments
``` bash
cd VQGAN_explanation
# Follow the guide in VQGAN_explanation/README.md
```
### 3.2 DALLE Experiments
``` bash
cd Dalle_explanation
# Follow the guide in Dalle_explanation/README.md
```