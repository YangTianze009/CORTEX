# CORTEX/Dalle_explanation

## Setup
1. Download the datasets and replace the `datasets` directory
(The dataset was generated using the [Dalle-mini](https://github.com/borisdayma/dalle-mini) model.)
2. Download pre-trained checkpoints and place them in the `checkpoints` directory

## Directory Structure

```
CORTEX/Dalle_explanation/
├── checkpoints/ # Model checkpoints
├── datasets/ # Datasets used for training/testing
├── bias_detection.py # Script for detecting model bias
├── dataset.py # Dataset loading and preprocessing
├── model.py # IEM architecture definition
├── README.md # Project description and usage instructions
├── test.py # Evaluation script
├── train.py # Training script for IEM
├── TIS_computation.py # Token Importance Score computation
├── TIS_analysis.py # Analysis of TIS results
```
The dataset was generated using the [Dalle-mini](https://github.com/borisdayma/dalle-mini) model.


## IEM Evaluation and Token Importance Score (TIS) Computation

This script provides a three-step evaluation process for our IEM model:

1. **Evaluate the classification accuracy of the IEM model**  
2. **Compute the Token Importance Score (TIS) for each token in every image**
3. **Evaluate the Bias based on TIS**

Run the following commands sequentially to complete the full evaluation:

```bash
# Step 1: Evaluate the IEM's classification accuracy
python test.py --model 1 --bias_type doctor_color  # or use doctor_gender

# Step 2: Compute Token Importance Score (TIS)
python TIS_computation.py --model 1 --bias_type doctor_color  # or use doctor_gender

# Step 3: Analysis the TIS
python TIS_analysis.py --model 1 --bias_type doctor_color # or use doctor_gender

# Step 4: Bias token detection
python bias_detection.py --bias_type doctor_color --model 1 --top_n 10 --token_num 10 # or use doctor_gender 
