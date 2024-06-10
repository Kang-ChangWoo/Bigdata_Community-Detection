import os
import numpy as np
import pandas as pd

# Function to read data from text files line by line and store in a dictionary
def read_data_from_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r') as file:
                    for line in file:
                        parts = line.split()
                        title = parts[0].strip('[]')
                        resolution = float(parts[2].split(',')[1])
                        nmi = float(parts[3].split(',')[1])
                        if title not in data:
                            data[title] = []
                        data[title].append(nmi)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    return df

# Function to save data to Excel
def save_data_to_excel(df, output_path):
    df.to_excel(output_path, index=False)

# Function to save data to CSV
def save_data_to_csv(df, output_path):
    df.to_csv(output_path, index=False)

# Main function to execute the process
def main(folder_path, output_excel_path, output_csv_path):
    df = read_data_from_files(folder_path)
    save_data_to_excel(df, output_excel_path)
    save_data_to_csv(df, output_csv_path)

# Specify the folder path and output file paths
# folder_path = '/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/test/TC1-all_including-GT(temp)'  # Change this to the path of your folder
#folder_path = '/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/real_trainset(temp)'  # Change this to the path of your folder
folder_path = '/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train/graph_dataset_v2(temp)'  # Change this to the path of your folder
output_excel_path = 'vis/output.xlsx'  # Change this to your desired Excel output path
output_csv_path = 'vis/output.csv'  # Change this to your desired CSV output path

# Run the main function
main(folder_path, output_excel_path, output_csv_path)
