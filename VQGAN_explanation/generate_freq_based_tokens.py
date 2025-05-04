import csv
from collections import defaultdict, Counter
from tqdm import tqdm 
import os

input_csv = 'datasets/train_embeddings.csv' 

label_token_counts = defaultdict(Counter)

with open(input_csv, 'r', encoding='utf-8') as csvfile:
    total_lines = sum(1 for _ in csvfile) - 1 
    csvfile.seek(0) 
    reader = csv.reader(csvfile)
    next(reader) 

    for row in tqdm(reader, total=total_lines, desc='Processing rows'):
        embedding = row[0]
        label = row[1]
        indices_str = row[2]

        indices_list = indices_str.strip('[]').split(', ')
        indices_list = [int(token) for token in indices_list if token]

        label_token_counts[label].update(indices_list)

for label, token_counter in tqdm(label_token_counts.items(), desc='Processing labels'):
    # Get the most common tokens
    top_tokens = token_counter.most_common(100)
    
    os.makedirs("results/TIS/frequency_baseline", exist_ok=True)
    # Create the output csv
    output_csv = f'results/TIS/frequency_baseline/label_{label}.csv'
    
    # Save to csv
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Token', 'Count'])
        for token, count in top_tokens:
            writer.writerow([token, count])

    print(f'Label {label}: Top tokens saved to {output_csv}')
