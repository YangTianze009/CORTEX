import torch
import os
import csv
from tqdm import tqdm
from dataset import get_test_dataloader
from model import ClassificationNet1
import argparse

def test_model(test_loader, num_classes, model_path, model_choice, device, bias_type):
    if model_choice == 1:
        model = ClassificationNet1(num_classes)
        model_name = 'ClassificationNet1'
    elif model_choice == 2:
        model = ClassificationNet2(num_classes)
        model_name = 'ClassificationNet2'
    elif model_choice == 3:
        model = ClassificationNet3(num_classes)
        model_name = 'ClassificationNet3'
    else:
        raise ValueError("Invalid model choice. Choose either 1, 2, or 3.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct_1 = 0

    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.topk(outputs, k=1, dim=1)
            total += labels.size(0)
            
            correct_1 += (predicted[:, :1] == labels.unsqueeze(1)).sum().item()


    top1_accuracy = 100 * correct_1 / total


    # Ensure results directory exists
    results_dir = f'results/{bias_type}/Classification/{model_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Write results to CSV file
    with open(f'{results_dir}/testing_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Accuracy'])
        writer.writerow([top1_accuracy])

    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a classification model.')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use for testing')
    parser.add_argument('--bias_type', type=str, default="doctor_gender",
                        help='Choose which bias type to train')
    args = parser.parse_args()

    test_csv = 'test_embeddings.csv'
    data_folder = f'datasets/{args.bias_type}/embeddings'
    batch_size = 64
    num_classes = 2  # Adjust according to the actual number of classes
    model_path = f"checkpoints/{args.bias_type}/ClassificationNet{args.model}/best_model.pth"

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    test_loader = get_test_dataloader(test_csv, data_folder, batch_size)
    test_model(test_loader, num_classes, model_path, args.model, device, args.bias_type)
