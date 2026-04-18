import os
import torch
import numpy as np
import faiss
import argparse
import time
import json
import random
import glob


def load_data_from_one(s_path, n_to_load=10000, perc = 0.01):
    # Load data from a sample under the form of a directory of .pt files 
    L_files = os.listdir(s_path)
    random.shuffle(L_files)
    data = []
    loaded=0
    tr=0
    while loaded < n_to_load and tr < len(L_files):
        if ".pt" in L_files[tr]:
            dat = torch.load(os.path.join(s_path,L_files[tr])).numpy()
        else:
            dat= np.load(os.path.join(s_path,L_files[tr]))
        if loaded+len(dat) > n_to_load:
            data.append(dat[:n_to_load-loaded])
            loaded+=len(dat[:n_to_load-loaded])
        else:
            data.append(dat)
            loaded+=len(dat)
        tr+=1
    data = np.concatenate(data)
    return data

def load_data_from_everywhere(list_of_paths, n_to_load=10000):
    data = []
    for s_path in list_of_paths:
        print("Loading data from ",s_path)
        data.append(load_data_from_one(str(s_path), n_to_load))
    data = np.concatenate(data)
    print("Total data loaded : ",data.shape)
    return data

def parse_list(s):
    # Parses a string representation of a list into an actual list
    l = s.strip("[]").split(",")
    return [int(x.strip()) for x in l]
            
def train_kmeans_faiss_multi_gpu(data, save_path, n_clusters_list, n_iter=20, verbose=True,min_points=32, max_points=1024):
    """
    Runs K-means using FAISS on multiple GPUs.
    
    Args:
        data (np.ndarray): The data array to cluster, shape (num_samples, num_features).
        n_clusters (int): The number of clusters to form.
        n_iter (int): Number of iterations for the K-means algorithm.
        verbose (bool): Whether to print the output during clustering.
        
    Returns:
        tuple: Cluster centroids and the assignments of each data point.
    """
    # Ensure data is in float32 format for FAISS
    data = data.astype(np.float32)
    
    # Get the number of available GPUs
    num_gpus = faiss.get_num_gpus()
    print(f"Using {num_gpus} GPUs for K-means clustering")

    # Create a resource object for each GPU
    res = [faiss.StandardGpuResources() for _ in range(num_gpus)]
    
    # Build a GPU index for clustering
    
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0  # Use GPU 0 as the primary device

    for n_clusters in n_clusters_list:
        print(f"Training K-means with {n_clusters} clusters")
        # Set up the K-means with FAISS using multi-GPU resources
        kmeans = faiss.Clustering(data.shape[1], n_clusters)
        kmeans.niter = n_iter
        kmeans.verbose = verbose
        kmeans.max_points_per_centroid = max_points
        kmeans.min_points_per_centroid = min_points

        # Create the multi-GPU index
        gpu_index = faiss.index_cpu_to_all_gpus(
            faiss.IndexFlatL2(data.shape[1]),  # Flat (L2) index
            co=None                            # Use all GPUs
        )
        # Train the K-means model using the multi-GPU index
        kmeans.train(data, gpu_index)

        # Get the cluster centroids and assignments for the input data
        centroids = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, -1)
        print("Saving centroids to ",os.path.join(save_path,'centroids_'+str(n_clusters)+'.npy'))
        np.save(os.path.join(save_path,'centroids_'+str(n_clusters)+'.npy'), centroids)
        gpu_index = faiss.index_gpu_to_cpu(gpu_index)
        # Save the CPU index
        print("Saving index to ",os.path.join(save_path,'index_'+str(n_clusters)+'.faiss'))
        faiss.write_index(gpu_index, os.path.join(save_path,'index_'+str(n_clusters)+'.faiss'))
        gpu_index.reset()  # Clear the GPU index to free memory


def train_clustering(path, save_path, n_clusters_list, n_iter, verbose, min_points, max_points, n_to_load=10000):
    os.makedirs(save_path, exist_ok=True)
    samples=[os.path.join(path,f) for f in os.listdir(path)]
    n_clusters_list = parse_list(n_clusters_list)
    np.random.shuffle(samples)
    tdeb=time.time()
    tepoch=time.time()
    data = load_data_from_everywhere(samples,n_to_load)
    print("Loading over :",time.time()-tdeb,"seconds")
    tload=time.time()
    train_kmeans_faiss_multi_gpu(data, save_path, n_clusters_list, n_iter, verbose, min_points, max_points)
    print("Training over :",time.time()-tload,"seconds")
    ttrain=time.time()
    print("Total time :",time.time()-tdeb,"seconds")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering using K-means with FAISS on multiple GPUs")
    parser.add_argument("data_path", type=str, help="Path to the directory of saved embeddings")
    parser.add_argument("save_path", type=str, help="Path to the directory where to save the clustering results")
    parser.add_argument("n_clusters_list", type=str, help="List of configs containing various numbers of clusters")
    parser.add_argument("n_iter", type=int, default=20, help="Number of iterations for the K-means algorithm")
    parser.add_argument("verbose", type=bool, default=True, help="Whether to print the output during clustering")
    parser.add_argument("min_points", type=int, default=32, help="Minimum number of points per cluster")
    parser.add_argument("max_points", type=int, default=1024, help="Maximum number of points per cluster (we recommand you settle it to the total number of points you will use divided by number of clusters unless it is too slow)")
    parser.add_argument("n_to_load", type=int, default=10000, help="Number of data points to use from each file")
    args = parser.parse_args()
    train_clustering(args.data_path, args.save_path, args.n_clusters_list, args.n_iter, args.verbose, args.min_points, args.max_points, args.n_to_load,)
