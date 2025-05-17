import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import pickle
import csv
import argparse
from tqdm import tqdm
import numpy as np
import random

def load_top_tokens(csv_path, filename, token_number):
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == filename:
                contributions = eval(row[2])
                sorted_indices = sorted(range(len(contributions)), key=lambda k: contributions[k], reverse=True)
                return sorted_indices[:token_number]
    return []

def preprocess_image(image, preprocess):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def calculate_logits_and_probs(models, image, target_label, token_list, target_token_indices, grid_size, device, preprocess):
    masked_image = image.copy()
    image_size = masked_image.size[0]
    patch_size = image_size // grid_size
    pixels = masked_image.load()

    target_tokens = [token_list[i] for i in target_token_indices if i < len(token_list)]
    
    for token_position in range(len(token_list)):
        if token_list[token_position] in target_tokens:
            row = token_position // grid_size
            col = token_position % grid_size
            for i in range(patch_size):
                for j in range(patch_size):
                    pixels[col * patch_size + i, row * patch_size + j] = (0, 0, 0)

    masked_batch = preprocess_image(masked_image, preprocess).to(device)
    masked_logits = {}
    masked_probs = {}
    with torch.no_grad():
        for name, model in models.items():
            masked_output = model(masked_batch)
            masked_logits[name] = masked_output[0, target_label].item()
            masked_probs[name] = torch.softmax(masked_output, dim=1)[0, target_label].item()

    return masked_logits, masked_probs

def main(args):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)  
    torch.manual_seed(random_seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = {
        'vit_b_16': torchvision.models.vit_b_16(pretrained=True),
        'vit_b_32': torchvision.models.vit_b_32(pretrained=True),
        'resnet18': torchvision.models.resnet18(pretrained=True),
        'resnet50': torchvision.models.resnet50(pretrained=True)
    }
    for model in models.values():
        model.eval()
        model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open('../datasets/test_token_indices.pkl', 'rb') as f:
        token_dict = pickle.load(f)

    image_base_path = '../datasets/images'
    test_csv = "../datasets/test_embeddings.csv"

    label_to_files = {i: [] for i in range(1000)}
    with open(test_csv, 'r', encoding='utf-8') as test_csvfile:
        reader = csv.reader(test_csvfile)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            label = int(row[1])
            label_to_files[label].append(filename)

    n_values = list(range(0, 55, 5))
    # n_values = list(range(0, 10, 5))
    print(n_values)
    results = {model_name: {n: {'original': {'logits': [], 'probs': []},
                                'target': {'logits': [], 'probs': []},
                                'random': {'logits': [], 'probs': []}}
                            for n in n_values}
               for model_name in models}

    for target_label in tqdm(range(1000), desc="Processing labels"):
        csv_path = f"../results/TIS/Net{args.model}/TIS_value_test/label_{target_label}.csv"

        npy_file_list = label_to_files[target_label]

        for npy_file in npy_file_list:
            subfolder, image_name = npy_file.split('_')
            image_name = image_name.replace('.npy', '.png')
            image_path = os.path.join(image_base_path, subfolder, image_name)
            
            if not os.path.exists(image_path):
                continue
            
            image = Image.open(image_path)
            token_list = token_dict.get(npy_file)
            
            if token_list is None:
                continue
            
            for n in n_values:
                if n == 0:
                    original_logits, original_probs = calculate_logits_and_probs(
                        models, image, target_label, token_list, [], 16, device, preprocess)
                    for model_name in models:
                        results[model_name][n]['original']['logits'].append(original_logits[model_name])
                        results[model_name][n]['original']['probs'].append(original_probs[model_name])
                else:
                    target_token_indices = load_top_tokens(csv_path, npy_file, n)
                    masked_logits_target, masked_probs_target = calculate_logits_and_probs(
                        models, image, target_label, token_list, target_token_indices, 16, device, preprocess)

                    random_token_indices = random.sample(range(len(token_list)), n)
                    masked_logits_random, masked_probs_random = calculate_logits_and_probs(
                        models, image, target_label, token_list, random_token_indices, 16, device, preprocess)

                    for model_name in models:
                        results[model_name][n]['target']['logits'].append(masked_logits_target[model_name])
                        results[model_name][n]['target']['probs'].append(masked_probs_target[model_name])
                        results[model_name][n]['random']['logits'].append(masked_logits_random[model_name])
                        results[model_name][n]['random']['probs'].append(masked_probs_random[model_name])

    # Calculate averages and differences
    average_results = []
    for model_name in models:
        original_logits = np.mean(results[model_name][0]['original']['logits'])
        original_probs = np.mean(results[model_name][0]['original']['probs'])
        
        average_results.append({
            'Model': model_name,
            'Num_Masked_Tokens': 0,
            'Method': 'original',
            'Avg_Logits': original_logits,
            'Avg_Probs': original_probs,
            'Logits_Diff': 0,
            'Probs_Diff': 0
        })
        
        for n in n_values[1:]:  # Skip 0 as it's already processed
            for method in ['target', 'random']:
                avg_logits = np.mean(results[model_name][n][method]['logits'])
                avg_probs = np.mean(results[model_name][n][method]['probs'])
                logits_diff = avg_logits - original_logits
                probs_diff = avg_probs - original_probs
                
                average_results.append({
                    'Model': model_name,
                    'Num_Masked_Tokens': n,
                    'Method': method,
                    'Avg_Logits': avg_logits,
                    'Avg_Probs': avg_probs,
                    'Logits_Diff': logits_diff,
                    'Probs_Diff': probs_diff
                })

    # Save results to CSV
    output_csv = f"sample_image_level_results.csv"
    evaluation_result_path = f"../results/Explanation/model_{args.model}"
    os.makedirs(evaluation_result_path, exist_ok=True)
    
    with open(os.path.join(evaluation_result_path, output_csv), 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Num_Masked_Tokens', 'Method', 'Avg_Logits', 'Avg_Probs', 'Logits_Diff', 'Probs_Diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in average_results:
            writer.writerow(row)

    print(f"Results saved to {os.path.join(evaluation_result_path, output_csv)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average logits and probability comparison for different models and token settings.")
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], required=True, help='Classification model to use')
    args = parser.parse_args()
    
    main(args)