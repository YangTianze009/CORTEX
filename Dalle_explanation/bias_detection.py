import csv
import ast
import pandas as pd
import os
import re
from typing import List, Dict, Tuple
from scipy import stats
import numpy as np
import argparse

def cliff_delta(x: List[float], y: List[float]) -> float:
    """
    Calculate Cliff's Delta.
    x, y: Python lists or NumPy arrays of numeric values.
    """
    x = np.array(x)
    y = np.array(y)
    Nx = len(x)
    Ny = len(y)
    total = 0.0
    for i in range(Nx):
        for j in range(Ny):
            if x[i] > y[j]:
                total += 1
            elif x[i] == y[j]:
                total += 0.5
    return (total - (Nx * Ny) / 2) / (Nx * Ny)

def load_top_n_tokens(csv_path: str, top_n: int, token_num: int) -> List[Tuple[str, List[str]]]:
    """
    Load multiple tokens from the Top N section of the specified CSV file.
    """
    tokens_data = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        in_top_n_section = False
        current_row = 0
        
        for row in reader:
            if f"Top {top_n} Tokens" in row:
                in_top_n_section = True
                continue
            
            if in_top_n_section and row and row[0] == "Token":
                continue
            
            if in_top_n_section and row:
                if current_row < token_num:
                    token = row[0]
                    files = row[2].split('; ')
                    tokens_data.append((token, files))
                    current_row += 1
                else:
                    break
    
    return tokens_data

def get_token_frequencies(token_list: List[int], data_csv_path: str) -> List[float]:
    """
    Get the total frequency of all tokens (as a group) in each file.
    """
    frequencies = []
    
    with open(data_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices = ast.literal_eval(row['indices'])
            # Count how many times any token from token_list appears in this file
            token_count = sum(1 for idx in indices if idx in token_list)
            frequencies.append(token_count)
    
    return frequencies

def perform_cliff(freqs1: List[float], freqs2: List[float]) -> Dict:
    """
    Perform Mann-Whitney U test between two groups of frequencies,
    and also compute Cliff's Delta as an effect size measure.
    """
    cd = cliff_delta(freqs1, freqs2)
    
    return {
        'cliff_delta': cd
    }

def analyze_token_statistics(tokens_data: List[Tuple[str, List[str]]], data_csv_path: str) -> Dict:
    """
    Analyze statistics for the specified tokens in the dataset.
    """
    token_list = [int(token) for token, _ in tokens_data]
    
    # Get frequencies for this group of tokens
    frequencies = get_token_frequencies(token_list, data_csv_path)
    
    # Calculate basic statistics
    total_files = len(frequencies)
    total_token_occurrences = sum(frequencies)
    
    stats = {
        'total_files': total_files,
        'total_token_occurrences': total_token_occurrences,
        'average_frequency_per_file': (total_token_occurrences / total_files) if total_files > 0 else 0,
        'number_of_target_tokens': len(token_list),
        'frequencies': frequencies 
    }
    
    return stats

def process_all_labels(input_folder: str, data_csv_path: str, output_folder: str,
                       top_n: int = 1, token_num: int = 50):
    """
    Process all label CSV files and save combined results.
    """
    # Create output folder for bias detection results
    os.makedirs(output_folder, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Store frequencies for Mann-Whitney U test
    label_frequencies = {}
    
    # Process each label file
    for filename in os.listdir(input_folder):
        if filename.startswith('label_') and filename.endswith('.csv'):
            # Extract label number
            label_num = re.search(r'label_(\d+)\.csv', filename)
            if not label_num:
                continue
            
            label_num = label_num.group(1)
            print(f"\nProcessing label {label_num}...")
            
            # Load and analyze tokens
            csv_path = os.path.join(input_folder, filename)
            tokens_data = load_top_n_tokens(csv_path, top_n, token_num)
            
            if not tokens_data:
                print(f"Failed to load tokens for label {label_num}")
                continue
            
            # Get statistics
            stats_dict = analyze_token_statistics(tokens_data, data_csv_path)
            stats_dict['label'] = label_num
            stats_dict['tokens'] = [token for token, _ in tokens_data]
            
            label_frequencies[label_num] = stats_dict['frequencies']
            del stats_dict['frequencies']
            
            all_results.append(stats_dict)
    
    if len(label_frequencies) >= 2:
        labels = sorted(label_frequencies.keys())
        for result in all_results:
            label1 = result['label']
            label2 = labels[1] if label1 == labels[0] else labels[0]
            # print(label1, label2) 
            test_results = perform_cliff(
                label_frequencies[label1],
                label_frequencies[label2]
            )
            
            # Add test results (including Cliff's Delta) to stats
            result['cliff_delta'] = test_results['cliff_delta']
            
            # Print results
            print(f"\nCliff delta results for label {label1}:")
            print(f"  Cliff's Delta:{test_results['cliff_delta']:.4f}")
    
    # Save combined results
    if all_results:
        combined_df = pd.DataFrame(all_results)
        output_filename = f'token_bias_top{top_n}_num{token_num}_cliff.csv'
        output_path = os.path.join(output_folder, output_filename)
    
        if 'cliff_delta' in combined_df.columns:
            combined_df['cliff_delta'] = combined_df['cliff_delta'].apply(lambda x: f'{x:.4f}')
        
        combined_df.to_csv(output_path, index=False)
        print(f"\nSaved combined statistics (including Cliff's Delta) to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process TIS statistics and perform bias detection.")
    parser.add_argument('--bias_type', type=str, choices=['doctor_color', 'doctor_gender'], default="doctor_color",
                        help='Type of bias to evaluate: doctor_color or doctor_gender')
    parser.add_argument('--model', type=int, default=1,
                        help='Model number, e.g., 1')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Top-N most important tokens to consider')
    parser.add_argument('--token_num', type=int, default=10,
                        help='Total number of tokens to analyze per image')

    args = parser.parse_args()

    # Configuration paths
    input_folder = f"results/{args.bias_type}/TIS/Net{args.model}/TIS_statistics_train"
    data_csv_path = f"datasets/doctor_original/information/results_none.csv"
    output_folder = f"results/{args.bias_type}/Bias_detection/"

    # Call processing function
    process_all_labels(input_folder, data_csv_path, output_folder, args.top_n, args.token_num)

if __name__ == "__main__":
    main()
