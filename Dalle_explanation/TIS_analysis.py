'''
This code is used to generate statistics information based on token activation value. Such as which token has the largest activation value and how many times it occurs in specific label
'''


import csv
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser(description="Visual Token contributions")
parser.add_argument('--gpu', type=int, default=0, help='Specify which GPU to use for computation')
parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                    help='Choose which model to test: 1 for ClassificationNet1, 2 for ClassificationNet2, or 3 for ClassificationNet3')
parser.add_argument('--bias_type', type=str, default="doctor_gender",
                    help='Choose which bias type to train')
args = parser.parse_args()

def load_token_indices(data_type, embedding_csv_path):
    token_indices_save_path = embedding_csv_path.replace(f"{data_type}_embeddings.csv", f"{data_type}_token_indices.pkl")
    
    if os.path.exists(token_indices_save_path):
        print(f"Loading token indices from {token_indices_save_path}")
        with open(token_indices_save_path, 'rb') as f:
            token_indices_dict = pickle.load(f)
        return token_indices_dict
    
    print(f"Processing token indices from {embedding_csv_path}")
    token_indices_dict = {}

    with open(embedding_csv_path, 'r') as infile:
        total_lines = sum(1 for _ in infile) - 1  

    with open(embedding_csv_path, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  
        for row in tqdm(reader, total=total_lines, desc="Loading token indices"):
            npy_file = row[0]
            token_indices = ast.literal_eval(row[2])
            token_indices_dict[npy_file] = token_indices
    
    with open(token_indices_save_path, 'wb') as f:
        pickle.dump(token_indices_dict, f)
    print(f"Token indices saved to {token_indices_save_path}")
    
    return token_indices_dict


def process_csv_files(folder_path, save_root, token_indices_dict, top_n_list=[1, 5, 10, 20]):
    os.makedirs(save_root, exist_ok=True)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    file_count = len(csv_files)
    print(f"file count is {file_count}")
    
    overall_token_statistics = {n: defaultdict(lambda: {"count": 0, "files": []}) for n in top_n_list}

    with tqdm(total=file_count, desc="Processing CSV files", unit="file") as pbar:
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            label_name = os.path.splitext(csv_file)[0]

            with open(csv_path, 'r') as infile:
                reader = csv.reader(infile)
                csv_data = list(reader)[1:] 

            label_token_statistics = {n: defaultdict(lambda: {"count": 0, "files": []}) for n in top_n_list}

            total_images = len(csv_data)

            for row in csv_data:
                npy_file = row[0]
                contribution_values = np.array(ast.literal_eval(row[2]))

                token_indices = token_indices_dict.get(npy_file)
                if token_indices is None:
                    continue

                for n in top_n_list:
                    top_n_indices = np.argsort(contribution_values)[-n:][::-1]  
                    top_tokens = [token_indices[i] for i in top_n_indices]
                    
                    for token in top_tokens:
                        label_token_statistics[n][token]["count"] += 1
                        label_token_statistics[n][token]["files"].append(npy_file)
                        overall_token_statistics[n][token]["count"] += 1
                        overall_token_statistics[n][token]["files"].append(npy_file)

            save_label_path = os.path.join(save_root, f"{label_name}.csv")
            with open(save_label_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow([f"Total Images: {total_images}"])

                for n in top_n_list:
                    writer.writerow([f"Top {n} Tokens"])
                    writer.writerow(["Token", "Frequency", "Files"])

                    top_tokens = sorted(label_token_statistics[n].items(), key=lambda x: x[1]["count"], reverse=True)[:100]
                    for token, data in top_tokens:
                        files_str = "; ".join(data["files"]) 
                        writer.writerow([token, data["count"], files_str])

            pbar.update(1)

    global_save_path = os.path.join(save_root, "global_top_tokens_statistics.csv")
    with open(global_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Top N", "Token", "Frequency", "Files"])

        for n in top_n_list:
            writer.writerow([f"Top {n} Tokens"])
            top_tokens = sorted(overall_token_statistics[n].items(), key=lambda x: x[1]["count"], reverse=True)[:100]
            for token, data in top_tokens:
                files_str = "; ".join(data["files"])
                writer.writerow([token, data["count"], files_str])


    print(f"Total files processed: {file_count}")


if __name__ == "__main__":

    bias_type = args.bias_type
    data_type = "train"
    folder_path = f"results/{bias_type}/TIS/Net{args.model}/TIS_value_{data_type}" 
    save_root = f"results/{bias_type}/TIS/Net{args.model}/TIS_statistics_{data_type}"  
    embedding_csv_path = f"datasets/{bias_type}/embeddings/{data_type}_embeddings.csv" 

    token_indices_dict = load_token_indices(data_type, embedding_csv_path)

    process_csv_files(folder_path, save_root, token_indices_dict)
