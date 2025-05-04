# IEM Evaluation and Token Importance Score (TIS) Computation

This script provides a two-step evaluation process for our IEM model:

1. **Evaluate the classification accuracy of the IEM model**  
2. **Compute the Token Importance Score (TIS) for each token in every image**

Run the following commands sequentially to complete the full evaluation:

```bash
# Step 1: Evaluate classification accuracy
python test.py --model 1 --bias_type doctor_color  # or use doctor_gender

# Step 2: Compute Token Importance Score (TIS)
python TIS_computation.py --model 1 --bias_type doctor_color  # or use doctor_gender

# Step 3: Analysis the TIS
python TIS_analysis.py --model 1 --bias_type doctor_color # or use doctor_gender

# Step 4: Bias token detection
python bias_detection.py --bias_type doctor_color --model 1 --top_n 10 --token_num 10 # or use doctor_gender 
