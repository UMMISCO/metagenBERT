import os
import torch
import numpy as np
import faiss
import argparse
import time
import json
import random
import glob

def assign_kmeans_faiss_multi_gpu(data, gpu_index, save_path,num_batch):
    """
    Assigns data points to clusters using a pre-trained K-means model.

    Args:
       	data (np.ndarray): The data array to cluster, shape (num_samples, num_features).
       	gpu_index (faiss.GpuMultipleCloner): Pre-trained K-means model.

    Returns:
       	np.ndarray: Assignments of each data point to a cluster.
    """
    # Assign data points to clusters
    _, assignments = gpu_index.search(data, 1)
    np.save(os.path.join(save_path,"assignments_"+str(num_batch)+".npy"), assignments)
    return assignments

def assign_by_batch(data_path, idx_path, gpu_index_list, save_path_list, nb_batch=200):
    for s in save_path_list:
        os.makedirs(os.path.join(s,data_path.split("/")[-1].split(".")[0]), exist_ok=True)
    files=[]
    for f in os.listdir(data_path):
        if isinstance(f, bytes):
            f = os.fsdecode(f)  # Convert bytes to string
        files.append(os.path.join(data_path,f))
    files.sort()
    idxs=[]
    for i in os.listdir(idx_path):
        if isinstance(i, bytes):
            i = os.fsdecode(i)
        idxs.append(os.path.join(idx_path,i))
    idxs.sort()
    try:
        assert np.array_equal([f.split("/")[-1].split("_")[-2:] for f in files],[i.split("/")[-1].split("_")[-2:] for i in idxs])
    except:
        print("The data and idx files in sample", data_path.split("/")[-1].split(".")[0], "do not match. Please check the data_path and idx_path.")
        return
    tr=0
    rnd = 0
    while tr < len(files):
        batch=[]
        batch_idx=[]
        n=0
        while n < nb_batch and tr < len(files):
            if ".pt" in files[tr]:
                batch.append(torch.load(files[tr]).numpy())
                batch_idx.append(torch.load(idxs[tr],map_location="cpu").cpu().numpy())
            else :
                batch.append(np.load(files[tr]))
                batch_idx.append(np.load(idxs[tr].cpu()))
            n+=1
            tr+=1
        batch = np.concatenate(batch)
        batch_idx = np.concatenate(batch_idx)
        rnd+=1
        try:
            assert batch.shape[0] == batch_idx.shape[0]
        except:
            print("The number of data points and indices in the batch do not match. Please check the data and idx files.")
            return
        for gpu_index, save_path in zip(gpu_index_list, save_path_list):
            assign_kmeans_faiss_multi_gpu(batch, gpu_index, os.path.join(save_path,data_path.split("/")[-1].split(".")[0]), rnd)
            np.save(os.path.join(save_path,data_path.split("/")[-1].split(".")[0],"idx_"+str(rnd)+".npy"), batch_idx)
        

def assign_all(save_path,data_path,idx_path,centroid_path_list,nb_batch=200):
    samples=sorted([os.path.join(data_path,f) for f in os.listdir(data_path)])
    idxs=sorted([os.path.join(idx_path,f) for f in os.listdir(idx_path)])
    try:
        assert np.array_equal([f.split("/")[-1].split(".")[0] for f in samples],[f.split("/")[-1].split(".")[0] for f in idxs])
    except:
        print("The samples and idx files do not match. Please check the data_path and idx_path.")
        return
    np.random.shuffle(samples)
    os.makedirs(save_path,exist_ok=True)
    centroid_path_list = centroid_path_list.strip("[]").split(",")
    centroids_list, gpu_index_list, save_path_list = [], [], []
    for c_path in centroid_path_list:
        centroids=np.load(c_path.strip().strip('"').strip("'"))
        k, d = centroids.shape
        gpu_res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, faiss.IndexFlatL2(d))
        gpu_index.add(centroids)
        centroids_list.append(centroids)
        gpu_index_list.append(gpu_index)
        save_path_list.append(os.path.join(save_path,c_path.split("/")[-1].split(".")[0]))
    for save_path in save_path_list:
        os.makedirs(save_path, exist_ok=True)
    tdeb =time.time()
    for (f, idx) in zip(samples, idxs):
        tsam=time.time()
        assign_by_batch(f,idx,gpu_index_list, save_path_list, nb_batch)
        print(f,"treated in",time.time()-tsam)
    print("dataset treated in",time.time()-tdeb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assigning using K-means with FAISS on multiple GPUs")
    parser.add_argument("save_path", type=str, help="Path to save the clustering results")
    parser.add_argument("data_path", type=str, help="Path to the data to cluster")
    parser.add_argument("idx_path", type=str, help="Path to the directory of saved indices corresponding to the data")
    parser.add_argument("centroid_path_list", type=str, help="List of paths of saved centroids, separated by commas, in a list format (e.g., '[\"path/to/centroid1.npy\", \"path/to/centroid2.npy\"]')")
    parser.add_argument("nb_batch", type=int, default=10, help="Number of batch files to process at once for assignment")
    args = parser.parse_args()
    assign_all(args.save_path, args.data_path, args.idx_path, args.centroid_path_list, args.nb_batch)