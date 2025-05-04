import torch
import os
import csv
from tqdm import tqdm
from dataset import get_test_dataloader
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3, ClassificationNet4, ClassificationNet4v2
import argparse

def test_model(test_loader, num_classes, model_choice, device):

    if model_choice == 1:
        model = ClassificationNet1(num_classes)
        model_name = 'ClassificationNet1'
    elif model_choice == 2:
        model = ClassificationNet2(num_classes)
        model_name = 'ClassificationNet2'
    elif model_choice == 3:
        model = ClassificationNet3(num_classes)
        model_name = 'ClassificationNet3'
    elif model_choice == 4:
        model = ClassificationNet4(num_classes)
        model_name = 'ClassificationNet4'
    else:
        raise ValueError("Invalid model choice.")
    model_path = f"/data2/ty45972_data2/taming-transformers/codebook_explanation_classification/checkpoints/generated_data/{model_name}/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    correct_10 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.topk(outputs, k=10, dim=1)
            total += labels.size(0)
            
            correct_1 += (predicted[:, :1] == labels.unsqueeze(1)).sum().item()
            correct_3 += (predicted[:, :3] == labels.unsqueeze(1)).sum().item()
            correct_5 += (predicted[:, :5] == labels.unsqueeze(1)).sum().item()
            correct_10 += (predicted[:, :10] == labels.unsqueeze(1)).sum().item()

    top1_accuracy = 100 * correct_1 / total
    top3_accuracy = 100 * correct_3 / total
    top5_accuracy = 100 * correct_5 / total
    top10_accuracy = 100 * correct_10 / total

    # Ensure results directory exists
    results_dir = f'results/Classification/{model_name}'
    os.makedirs(results_dir, exist_ok=True)

    # Write results to CSV file
    with open(f'{results_dir}/testing_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Top-10 Accuracy'])
        writer.writerow([top1_accuracy, top3_accuracy, top5_accuracy, top10_accuracy])

    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%')
    print(f'Top-3 Accuracy: {top3_accuracy:.2f}%')
    print(f'Top-5 Accuracy: {top5_accuracy:.2f}%')
    print(f'Top-10 Accuracy: {top10_accuracy:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a classification model.')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], default=1,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    parser.add_argument('--gpu', type=int, default=1,
                        help='Specify which GPU to use for testing')
    args = parser.parse_args()

    test_csv = 'test_embeddings.csv'
    data_folder = 'datasets/'
    batch_size = 64
    num_classes = 1000  # Adjust according to the actual number of classes
    

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    test_loader = get_test_dataloader(test_csv, data_folder, batch_size)
    test_model(test_loader, num_classes, args.model, device)
