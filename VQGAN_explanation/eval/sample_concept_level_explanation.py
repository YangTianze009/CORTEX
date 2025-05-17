import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import pickle
import csv
import argparse
from tqdm import tqdm

def load_top_tokens(csv_path, top_n, token_number):
    target_token_list = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        in_top_n_section = False
        current_row = 0

        for row in reader:
            if f"Top {top_n} Tokens" in row:
                in_top_n_section = True
                next(reader)
                current_row = 0
                continue

            if in_top_n_section:
                if "Top" in row[0] or current_row == token_number:
                    break
                token = int(row[0])
                target_token_list.append(token)
                current_row += 1

    return target_token_list

def load_top_tokens_baseline(input_csv, top_n):
    tokens = []
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            token = int(row['Token'])
            count = int(row['Count'])
            tokens.append((token, count))
    
    tokens.sort(key=lambda x: x[1], reverse=True)
    top_n_tokens = [token for token, _ in tokens[:top_n]]
    
    return top_n_tokens

def preprocess_image(image, preprocess):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def calculate_logits_diff(models, image, target_label, token_list, target_token_list, grid_size, device, preprocess):
    # Original image prediction
    input_batch = preprocess_image(image, preprocess).to(device)
    original_logits = {}
    original_probs = {}
    with torch.no_grad():
        for name, model in models.items():
            original_output = model(input_batch)
            original_logits[name] = original_output[0, target_label].item()
            original_probs[name] = torch.nn.functional.softmax(original_output, dim=1)[0, target_label].item()
    
    # Mask token areas and predict logits
    masked_image = image.copy()
    image_size = masked_image.size[0]
    patch_size = image_size // grid_size
    pixels = masked_image.load()
    
    for token_position in range(len(token_list)):
        if token_list[token_position] in target_token_list:
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
            masked_probs[name] = torch.nn.functional.softmax(masked_output, dim=1)[0, target_label].item()

    return {
        name: {
            'logits_diff': original_logits[name] - masked_logits[name],
            'original_prob': original_probs[name],
            'masked_prob': masked_probs[name]
        } for name in models
    }

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up all models
    models = {
        'vit_b_16': torchvision.models.vit_b_16(pretrained=True),
        'vit_b_32': torchvision.models.vit_b_32(pretrained=True),
        'resnet18': torchvision.models.resnet18(pretrained=True),
        'resnet50': torchvision.models.resnet50(pretrained=True)
    }
    for model in models.values():
        model.eval()
        model.to(device)

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load token dictionary
    with open('../datasets/test_token_indices.pkl', 'rb') as f:
        token_dict = pickle.load(f)

    # Define paths
    image_base_path = '../datasets/images'
    test_csv = "../datasets/test_embeddings.csv"

    # Read test CSV once and organize data
    label_to_files = {i: [] for i in range(1000)}
    with open(test_csv, 'r', encoding='utf-8') as test_csvfile:
        reader = csv.reader(test_csvfile)
        next(reader)  # Skip header
        for row in reader:
            filename = row[0]
            label = int(row[1])
            label_to_files[label].append(filename)

    # Initialize results dictionary
    results = {model_name: {
        'target': {'logits_diff': 0, 'masked_prob': 0},
        'baseline': {'logits_diff': 0, 'masked_prob': 0},
        'original_prob': 0,
        'count': 0
    } for model_name in models}

    # Main loop for processing all labels
    for target_label in tqdm(range(1000), desc="Processing labels"):
        csv_path = f"../results/TIS/Net{args.model}/TIS_statistics_train/label_{target_label}.csv"
        baseline_path = f"../results/TIS/frequency_baseline/label_{target_label}.csv"

        # Load tokens
        target_token_list = load_top_tokens(csv_path, args.top_n, args.token_num)
        target_token_list_baseline = load_top_tokens_baseline(baseline_path, args.token_num)

        # Get file list for the current label
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
            
            diff_target = calculate_logits_diff(models, image, target_label, token_list, target_token_list, 16, device, preprocess)
            diff_baseline = calculate_logits_diff(models, image, target_label, token_list, target_token_list_baseline, 16, device, preprocess)
            
            for model_name in models:
                results[model_name]['target']['logits_diff'] += diff_target[model_name]['logits_diff']
                results[model_name]['target']['masked_prob'] += diff_target[model_name]['masked_prob']
                results[model_name]['baseline']['logits_diff'] += diff_baseline[model_name]['logits_diff']
                results[model_name]['baseline']['masked_prob'] += diff_baseline[model_name]['masked_prob']
                results[model_name]['original_prob'] += diff_target[model_name]['original_prob']
                results[model_name]['count'] += 1

    # Calculate averages and prepare output
    output_data = []
    for model_name, result in results.items():
        count = result['count']
        if count > 0:
            avg_diff_target = result['target']['logits_diff'] / count
            avg_diff_baseline = result['baseline']['logits_diff'] / count
            avg_original_prob = result['original_prob'] / count
            avg_masked_prob_target = result['target']['masked_prob'] / count
            avg_masked_prob_baseline = result['baseline']['masked_prob'] / count
            output_data.append([model_name, avg_diff_target, avg_diff_baseline, avg_original_prob, avg_masked_prob_target, avg_masked_prob_baseline])

    # Calculate overall averages
    overall_avg = [sum(row[i] for row in output_data) / len(output_data) for i in range(1, len(output_data[0]))]
    output_data.append(['Overall'] + overall_avg)

    # Write results to CSV
    output_csv = f"sample_concept_level_results_top{args.top_n}_token{args.token_num}.csv"
    evaluation_result_path = f"../results/Explanation/model_{args.model}"
    os.makedirs(evaluation_result_path, exist_ok=True)
    with open(os.path.join(evaluation_result_path, output_csv), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Avg_Diff_Target', 'Avg_Diff_Baseline', 'Avg_Original_Prob', 'Avg_Masked_Prob_Target', 'Avg_Masked_Prob_Baseline'])
        writer.writerows(output_data)

    print(f"Results saved to {os.path.join(evaluation_result_path, output_csv)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logits difference and probability comparison for different models and token settings.")
    parser.add_argument('--top_n', type=int, default=20, help="Top N tokens to consider")
    parser.add_argument('--token_num', type=int, default=100, help="Number of tokens to use")
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], required=True, help='Classification model to use')
    args = parser.parse_args()
    
    main(args)