import torch
import numpy as np
import csv
from tqdm import tqdm
from dataset import get_test_dataloader, get_train_val_dataloaders
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3
import argparse
import os
import random

class AverageSaliency:
    def __init__(self, model, output_index=0):
        self.model = model
        self.output_index = output_index

    def get_grad(self, input_images, target_indices):
        input_images.requires_grad = True
        outputs = self.model(input_images)
        self.model.zero_grad()

        # For each sample in the batch, we only compute gradients for its own target class
        gradients = []
        for i, index in enumerate(target_indices):
            grad_output = torch.zeros_like(outputs)
            grad_output[i, index] = 1.0  # Only compute gradient w.r.t. the target index for this sample
            grad = torch.autograd.grad(
                outputs=outputs, inputs=input_images, grad_outputs=grad_output, create_graph=True
            )[0]
            gradients.append(grad[i].unsqueeze(0))  # Save the gradient for this sample only

        return torch.cat(gradients, dim=0)  # Return all gradients as a batch

    def get_average_grad(self, input_images, target_indices, stdev_spread=0.2, nsamples=20):
        stdev = stdev_spread * (torch.max(input_images) - torch.min(input_images))

        total_gradients = torch.zeros_like(input_images, dtype=torch.float64)
        for _ in range(nsamples):
            noise = torch.normal(mean=0, std=stdev, size=input_images.shape).to(input_images.device)
            noisy_images = input_images + noise

            gradients = self.get_grad(noisy_images, target_indices)
            total_gradients += gradients

        return total_gradients / nsamples

class SingleSaliency(AverageSaliency):
    def __init__(self, model, output_index=0):
        super(SingleSaliency, self).__init__(model, output_index)

    def get_grad(self, input_images, target_indices):
        return super().get_grad(input_images, target_indices)

def compute_contribution(test_loader, model_path, device, results_dir, selected_labels, num_classes):  # Adjust according to your actual number of classes
    if args.model == 1:
        model = ClassificationNet1(num_classes)
    elif args.model == 2:
        model = ClassificationNet2(num_classes)
    elif args.model == 3:
        model = ClassificationNet3(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(results_dir, exist_ok=True)

    for idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Computing Saliency Maps")):
        inputs, labels, filenames = inputs.to(device), labels.to(device), test_loader.dataset.data[idx * test_loader.batch_size: (idx + 1) * test_loader.batch_size]

        # Only process samples with labels in selected_labels
        mask = torch.tensor([label in selected_labels for label in labels])
        inputs = inputs[mask]
        labels = labels[mask]
        filenames = [filenames[i] for i in range(len(filenames)) if mask[i]]

        if inputs.size(0) == 0:
            continue  # Skip batch if no valid samples

        # Create saliency maps for the entire batch
        single_saliency = SingleSaliency(model)
        grads = single_saliency.get_average_grad(inputs, labels).detach().cpu().numpy()

        # For each sample in the batch, save the gradient result
        for i, label in enumerate(labels):
            csv_path = os.path.join(results_dir, f"label_{label}.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow(['filename', 'label', 'contribution'])  # Write header if new file
            
            with open(csv_path, 'a', newline='') as f:
                csvwriter = csv.writer(f)
                avg_grad_flat = grads[i].mean(axis=0).flatten()  # Flatten and average over the 256-dim embeddings
                max_grad_flat = grads[i].max(axis=0).flatten()

                # Write results to the CSV after processing each image
                if args.method == "max":
                    csvwriter.writerow([filenames[i], label.item(), max_grad_flat.tolist()])
                elif args.method == "ave":
                    csvwriter.writerow([filenames[i], label.item(), avg_grad_flat.tolist()])

    print(f'Token contributions saved to individual label CSVs in {results_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute saliency maps for a classification model.')
    parser.add_argument('--method', type=str, default="max", help='Method for contribution calculation: "max" or "ave"')
    parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                        help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
    parser.add_argument('--bias_type', type=str, default="doctor_gender",
                        help='Choose which bias type to train')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Choose the batch size for TIS computation')
    args = parser.parse_args()

    data_type = "train"

    selected_labels = [0, 1]
    num_classes = 2
    print(f"Selected labels: {selected_labels}")

    test_csv = f'{data_type}_embeddings.csv'
    val_csv = 'validation_embeddings.csv'
    
    data_folder = f'datasets/{args.bias_type}/embeddings'
    batch_size = args.batch_size
    model_path = f"checkpoints/{args.bias_type}/ClassificationNet{args.model}/best_model.pth"
    results_dir = f"results/{args.bias_type}/TIS/Net{args.model}/TIS_value_{data_type}"
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # to generate TIS on training data
    if data_type == "train":
        test_loader, val_loader = get_train_val_dataloaders(test_csv, val_csv, data_folder, batch_size, shuffle=False)
    elif data_type == "test":
        test_loader = get_test_dataloader(test_csv, data_folder, batch_size)
    compute_contribution(test_loader, model_path, device, results_dir, selected_labels, num_classes)
