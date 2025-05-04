# CORTEX/VQGAN_explanation

This subfolder contains the implementation of CORTEX to explain the VQGAN model.

## Directory Structure

```
CORTEX/VQGAN_explanation/
├── checkpoints/            # Model checkpoints (download required)
├── datasets/               # Datasets (download required)
├── eval/                   # Evaluation scripts
│   ├── codebook_level_explanation.py
│   ├── sample_concept_level_explanation.py
│   ├── sample_image_level_explanation.py
├── explanation_evaluation/ # Evaluation metrics for explanations
├── logs/                   # Training logs
├── results/                # Results directory
├── taming/                 # VQGAN related modules
├── model.py                # IEM architecture
├── new_vqgan.py            # Prepare for the VQGAN repository
├── dataset.py              # Dataset loader
├── train.py                # Training script for IEM
├── test.py                 # Evaluation script
├── TIS_computation.py      # Token Importance Score computation
├── TIS_analysis.py         # TIS analysis for concept-level explanations
├── generate_freq_based_tokens.py # Generate frequency-based baseline
```

## Setup
1. Clone the repository of [VQGAN](https://github.com/CompVis/taming-transformers)
2. Place the new_vqgan.py file into the VQGAN repository under the taming-transformers/taming/models directory
3. Download the datasets and replace the `datasets` directory
(The dataset was generated using the [VQGAN](https://github.com/CompVis/taming-transformers model.))
4. Download pre-trained checkpoints and place them in the `checkpoints` directory

## 1. Training

You can train also your own Interpretable Explanation Model (IEM) on different Vector-Quantized Generative Models (VQGMs).

### Input Format
Note that the model's input is  the token-based embedding with dimensions (256, 16, 16).

### Training Command
```bash
python train.py --model {model_name}
```
where `model_name` can be 1, 2, 3, or 4.

## 2. Evaluation Preparation

### 2.1 Test IEM Classification Performance

To evaluate the classification performance of your trained IEM:

```bash
python test.py --model {model_name}
```
where `model_name` can be 1, 2, 3, or 4.

### 2.2 Compute Token Importance Scores (TIS)

Generate Token Importance Scores for training or testing data:

```bash
python TIS_computation.py --model {model_name} --data_type {data_type} --batch_size {batch_size} --gpu {gpu_number}
```

Parameters:
- `model_name`: 1, 2, 3, or 4
- `data_type`: 
  - `train`: generates data for `eval/sample_concept_level_explanation.py`
  - `test`: generates data for `eval/sample_image_level_explanation.py`
- `batch_size`: Integer value (recommended value depends on your GPU memory)
- `gpu_number`: GPU device number (integer)

Example:
```bash
python TIS_computation.py --model 1 --data_type train --batch_size 25 --gpu 1
```

Note: This process may take a considerable amount of time.

### 2.3 Generate Frequency-based Baseline

```bash
python generate_freq_based_tokens.py
```

### 2.4 Generate Sample Concept-level Tokens

```bash
python TIS_analysis.py --model {model_name}
```

## 3. Evaluation

### 3.1 Navigate to the Evaluation Directory

```bash
cd eval
```

### 3.2 Sample Image-level Explanation

To evaluate image-level explanations:

```bash
python sample_image_level_explanation.py --model {model_name}
```

### 3.3 Sample Concept-level Explanation

To evaluate concept-level explanations:

```bash
python sample_concept_level_explanation.py --model {model_name} --top_n {top-n value} --token_num {token number}
```

Parameters:
- `model_name`: 1, 2, 3, or 4
- `top_n`: Select the top-n tokens with the highest activation frequency for each image
- `token_num`: Number of tokens to consider

### 3.4 Codebook-level Explanation

To generate codebook-level explanations:

Place the cloned VQGAN repository directory path in the codebook_level_explanation.py file by replacing `VQGAN_directory = {Your VQGAN directory}` with your actual path, so that you can use VQGAN's decoder to test the optimization performance.

```bash
python codebook_level_explanation.py --model {model_name} --steps {optimization_steps} --lr {learning_rate} --optimization_type {token_selection or embedding}
```

Example with some default settings:
```bash
python codebook_level_explanation.py --model 1 --optimization_type token
```

## License

[License information]

## Acknowledgements

[Acknowledgements information]
