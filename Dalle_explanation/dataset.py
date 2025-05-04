import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_file, data_folder):
        self.data_folder = data_folder
        self.data = []
        self.labels = []

        with open(embeddings_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(',')
                filename = parts[0]  
                label = int(parts[1])  
                self.data.append(filename) 
                self.labels.append(label)  

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):
        file_path = f"{self.data_folder}/{self.data[idx]}" 
        embedding = np.load(file_path)  
        label = self.labels[idx] 
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  

def get_train_val_dataloaders(train_csv, val_csv, data_folder, batch_size, shuffle=True):
    train_dataset = EmbeddingDataset(f"{data_folder}/" + train_csv, f"{data_folder}/train")  
    val_dataset = EmbeddingDataset(f"{data_folder}/" + val_csv, f"{data_folder}/val")  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

    return train_loader, val_loader  

def get_test_dataloader(test_csv, data_folder, batch_size):
    test_dataset = EmbeddingDataset(f"{data_folder}/" + test_csv, f"{data_folder}/test") 

    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader 
