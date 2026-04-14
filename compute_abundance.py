import argparse
import os
import torch
import numpy as np
import json
import multiprocessing

def get_abundance(sample, nb_clust):
    print(sample)
    if "abundance.npy" in os.listdir(sample):
        print(sample,"already treated")
    else:
        abundance = np.zeros((nb_clust))
        assigned = np.load(sample+"/assignments.npy")
        for read in assigned:
            ele = read[0]
            abundance[ele] += 1
        total_assigned = np.sum(abundance)
        abundance = abundance / total_assigned
        np.save(sample + "/abundance.npy", abundance)
        print(sample,"saved")
        return abundance

def all_samples_parallel(samples_dir, nb_clust, num_workers=multiprocessing.cpu_count()):
    samples = os.listdir(samples_dir)
    samples.sort()
    with multiprocessing.Pool(processes=num_workers) as pool:
        for sample in samples:
            sample = os.path.join(samples_dir, sample)
            pool.apply_async(get_abundance, args=(sample, nb_clust))
        pool.close()
        pool.join()
        print("All samples in dataset "+ samples_dir + " have been processed.")



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Compute abundance for each sample in the dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory.")
    parser.add_argument("number", type=int, help="Number of clusters to compute abundance for.")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes to use for computing abundance.")
    args = parser.parse_args()
    all_samples_parallel(args.dataset_path, args.number, args.num_workers)

