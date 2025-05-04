import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from dataset import get_train_val_dataloaders
from model import ClassificationNet1, ClassificationNet2, ClassificationNet3, ClassificationNet4
from tqdm import tqdm
from datetime import datetime
import argparse

def train_model(train_loader, val_loader, num_classes, num_epochs, learning_rate, step_size, gamma, model_choice):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")

    if model_choice == 1:
        model = ClassificationNet1(num_classes).to(device)
        model_name = 'ClassificationNet1'
    elif model_choice == 2:
        model = ClassificationNet2(num_classes).to(device)
        model_name = 'ClassificationNet2'
    elif model_choice == 3:
        model = ClassificationNet3(num_classes=1000).to(device)
        model_name = 'ClassificationNet3'
    elif model_choice == 4:
        model = ClassificationNet4(num_classes=1000).to(device)
        model_name = 'ClassificationNet4'
    else:
        raise ValueError("Invalid model choice. Choose either 1 or 2 or 3.")

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Create checkpoints and logs folders
    checkpoint_dir = f'checkpoints/{model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(f'logs/'):
        os.makedirs(f'logs/')

    # Create a log file with a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    log_file_path = f'logs/training_log_{timestamp}.txt'
    best_val_loss = float('inf')
    
    with open(log_file_path, 'w') as log_file:
        for epoch in range(num_epochs):
            print(f"Training Epoch {epoch+1}/{num_epochs}")
            model.train()
            running_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc="Training")
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # print(f"output is {outputs[0]}")
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf detected in loss at epoch {epoch}, batch {train_loader_tqdm.n}")
                    continue
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1))

            avg_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}\n")

            print(f"Validating Epoch {epoch+1}/{num_epochs}")
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            val_loader_tqdm = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for inputs, labels in val_loader_tqdm:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    val_loader_tqdm.set_postfix(loss=val_loss / (val_loader_tqdm.n + 1))

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            log_file.write(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')
            log_file.flush()

            # Save the model for each epoch
            torch.save(model.state_dict(), f'{checkpoint_dir}/model.pth')

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{checkpoint_dir}/best_model.pth')

            # Step the scheduler
            scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a classification model.')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], default=4,
                        help='Choose which model to train: 1 for ClassificationNet1 or 2 for ClassificationNet2 or 3 for ClassificationNet3')
    args = parser.parse_args()

    train_csv = 'train_embeddings.csv'
    val_csv = 'validation_embeddings.csv'
    data_folder = 'datasets/'
    batch_size = 512
    num_classes = 1000  # Adjust according to the actual number of classes
    num_epochs = 80
    learning_rate = 0.001
    step_size = 20  # Decay learning rate every step_size epochs
    gamma = 0.1  # Multiply learning rate by gamma at each stepß

    train_loader, val_loader = get_train_val_dataloaders(train_csv, val_csv, data_folder, batch_size)
    train_model(train_loader, val_loader, num_classes, num_epochs, learning_rate, step_size, gamma, args.model)
