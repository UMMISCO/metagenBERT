import os
import numpy as np
import multiprocessing as mp
import argparse

def purge_sample(path):
    print(f"Purging sample in {path}...")
    assigns  = [f for f in os.listdir(path) if "assignments_" in f]
    idx = [f for f in os.listdir(path) if "idx_" in f]
    if len(assigns)==0:
        print(f"{path} already treated or empty")
    else :
        assignments = np.concatenate([np.load(os.path.join(path, f)) for f in assigns])
        indexes = np.concatenate([np.load(os.path.join(path, f)) for f in idx])
        _ , unique_pos = np.unique(indexes, return_index=True)
        filtered_indexes = indexes[unique_pos]
        filtered_assignments = assignments[unique_pos]
        np.save(os.path.join(path, "assignments.npy"), filtered_assignments)
        np.save(os.path.join(path, "indexes.npy"), filtered_indexes)
        print(f"Over for {path}")
        for f in assigns + idx:
            os.remove(os.path.join(path, f))

def purge_dataset(dataset_path, num_workers=mp.cpu_count()):
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset path {dataset_path} does not exist.")
    for sample in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, sample)):
            raise ValueError(f"The dataset path {dataset_path} should contain only directories for each sample.")
    sample_paths = [os.path.join(dataset_path, sample) for sample in os.listdir(dataset_path)]
    with mp.Pool(processes=num_workers) as pool:
        pool.map(purge_sample, sample_paths)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge clusters from a dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the assignment directory.")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of worker processes to use for purging.")
    args = parser.parse_args()
    purge_dataset(args.dataset_path, args.num_workers)
    print(f"Purged clusters in dataset at {args.dataset_path}.")
