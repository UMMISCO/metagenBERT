import numpy as np
import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import AutoConfig
import time
import os
from torch.utils.data import Dataset, DataLoader
import gc
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import faiss

# This custom Dataset class keeps track of the index of each sequence in the fasta file (these indexes can be used to reconstruct each cluster 
# obtained through MetagenBERT by fetching the sequences in the original fasta file)

class SentenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.sentences = self._load_sentences(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_sentences(self, file_path):
        with open(file_path, 'r') as f:
            sentences = [line.strip() for line in f]
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_input = self.tokenizer(sentence, padding="max_length", truncation=True, max_length=self.max_length)
        encoded_input['idx'] = idx
        return encoded_input

# ------------------------------
# Load FAISS indexes
# ------------------------------
def load_faiss_indexes(centroids_paths, gpu_id):
    """
    Load centroid files under directories in a directory "centroids_dirs".
    Returns: dict {index_name: faiss_index}
    """
    centroids_paths = centroids_paths.strip("[]").split(",")
    index_dict = {}
    for cent_path in centroids_paths:
        if not os.path.exists(cent_path):
            continue

        centroids = np.load(cent_path).astype("float32")
        dim = centroids.shape[1]

        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(dim)
        faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, index_flat)
        faiss_index.add(centroids)

        index_dict[cent_path] = faiss_index
        print(f"[GPU {torch.cuda.get_device_name(gpu_id)}] Loaded FAISS index for centroids from {cent_path} with {centroids.shape[0]} centroids and dimension {dim}.")

    return index_dict
    
def embed_assign(rank, model_path,sequence_dir,max_length,saving_path,batch_size=10000, world_size=1, centroids_paths=None):
    #Create all saving paths : one directory for assignments and one for indexes. In each, one directory per sample analyzed
    if rank == 0:
        L_files = os.listdir(sequence_dir)
        L_files.sort()
        assign_saving_path = os.path.join(saving_path,"assignments")
        idx_saving_path = os.path.join(saving_path,"idx")
        os.makedirs(assign_saving_path, exist_ok=True)
        os.makedirs(idx_saving_path, exist_ok=True)
        for f in os.listdir(sequence_dir):
            os.makedirs(os.path.join(idx_saving_path,f), exist_ok=True)
        for centroid_path in centroids_paths.strip("[]").split(","):
            os.makedirs(os.path.join(assign_saving_path,centroid_path.split("/")[-1].split(".")[0]), exist_ok=True)
            for f in L_files:
                os.makedirs(os.path.join(saving_path,"assignments",centroid_path.split("/")[-1].split(".")[0],f.split("/")[-1].split(".")[0]), exist_ok=True)
    gpu = torch.device("cuda")

    ## Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Initiating parallelization            
    dist.init_process_group(backend='nccl',
                        init_method="tcp://127.0.0.1:12355", 
                        world_size= world_size, 
                        rank=rank)
    torch.cuda.set_device(rank)
    model = model.to(gpu)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DDP(model, device_ids=[rank])


    # Loop on every file in sequence_dir
    model.eval()
    L_files = os.listdir(sequence_dir)
    L_files.sort()

    faiss_indexes = load_faiss_indexes(centroids_paths, gpu_id=rank)
    for sequence_file in L_files:
        print("Starting assignment of sample "+sequence_file.split("/")[-1].split(".")[0]+"on GPU "+str(torch.cuda.get_device_name(gpu_id)))
        sequence_file = os.path.join(sequence_dir, sequence_file)
        idx_file_saving_path = os.path.join(saving_path,"idx",sequence_file.split("/")[-1].split(".")[0])
        batch_index=0
        # Define datasets
        dataset = SentenceDataset(sequence_file, tokenizer, max_length)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length",max_length=max_length)
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                num_replicas=world_size,
                                                                rank=rank,
                                                                shuffle=False)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=data_sampler,
                                            collate_fn=data_collator)
        with torch.no_grad():  # Disable gradient calculation for inference
            for batch in data_loader:
                # Tokenize the batch
                gpu_batch = {k: v.to(gpu) for k, v in batch.items()}
                batch_embeddings = model(**gpu_batch)[0]
                for centroid_path, faiss_index in faiss_indexes.items():  
                    _, assignments = faiss_index.search(torch.mean(batch_embeddings,dim=1).cpu().numpy().astype("float32"), 1)
                    np.save(os.path.join(saving_path,"assignments",centroid_path.split("/")[-1].split(".")[0],sequence_file.split("/")[-1].split(".")[0],"assignments_"+str(batch_index)+"_"+str(rank)+".npy"), assignments)
                # Save assignments and indexes
                np.save(os.path.join(idx_file_saving_path,"idx_"+str(batch_index)+"_"+str(rank)+".npy"), batch["idx"].cpu().numpy())
                ## In case of GPU memory overloading, activate garbage collect
                #del gpu_batch
                #del batch_embeddings
                #torch.cuda.empty_cache()
                #gc.collect()
                batch_index+=1

def main(args):
    world_size = args.world_size
    mp.spawn(embed_assign, args=(args.model_path,args.sequence_dir,args.max_length,args.saving_path,args.batch_size,args.world_size,args.centroids_paths), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cleaned_Embed sequences using a pretrained model.")
    # Add an argument for the directory path
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("sequence_dir", type=str, help="Sequences directory of files to embed file")
    parser.add_argument("max_length", type=int, help="Length to which each sequence will be truncated (for short reads and considering the DNABERT-2 tokenizer for example, it can be as low as 60 tokens)")
    parser.add_argument("saving_path", type=str, help="Where to save the embedded sequences")
    parser.add_argument("batch_size", type=int, default=10000, help="Batch size for embedding")
    parser.add_argument("world_size", type=int, help="Number of processes")
    parser.add_argument("centroids_paths", type=str, help="Path to the file of centroids to assign to, separated by commas, in a list format (e.g., '[\"path/to/centroid_dir1\", \"path/to/centroid_dir2\"]')")
    args = parser.parse_args()
    main(args)


