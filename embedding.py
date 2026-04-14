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
    
def embed(rank, model_path,sequence_dir,max_length,saving_path,batch_size=10000, world_size=1):
    #Create all saving paths : one directory for embeddings and for indexes. In each, one directory per sample analyzed
    if rank == 0:
        emb_saving_path = os.path.join(saving_path,"embeddings")
        idx_saving_path = os.path.join(saving_path,"idx")
        os.makedirs(emb_saving_path, exist_ok=True)
        os.makedirs(idx_saving_path, exist_ok=True)
        for f in os.listdir(sequence_dir):
            os.makedirs(os.path.join(emb_saving_path,f), exist_ok=True)
            os.makedirs(os.path.join(idx_saving_path,f), exist_ok=True)
            
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
    for sequence_file in L_files:
        batch_index=0
        sequence_file = os.path.join(sequence_dir, sequence_file)
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
        emb_file_saving_path = os.path.join(saving_path,"embeddings",sequence_file.split("/")[-1])
        idx_file_saving_path = os.path.join(saving_path,"idx",sequence_file.split("/")[-1])
        with torch.no_grad():  # Disable gradient calculation for inference
            for batch in data_loader:
                # Tokenize the batch
                gpu_batch = {k: v.to(gpu) for k, v in batch.items()}
                batch_embeddings = model(**gpu_batch)[0]
                # Save embeddings and indexes
                torch.save(torch.mean(batch_embeddings,dim=1).cpu(), os.path.join(emb_file_saving_path, f'embeddings_{batch_index}_{rank}.pt'))
                torch.save(gpu_batch['idx'],os.path.join(idx_file_saving_path,f'idx_{batch_index}_{rank}.pt'))
                ## In case of GPU memory overloading, activate garbage collect
                #del gpu_batch
                #del batch_embeddings
                #torch.cuda.empty_cache()
                #gc.collect()
                batch_index+=1
        print(f"Process {rank} finished embedding for file {sequence_file}.")
    print(f"Process {rank} finished embedding.")

def main(args):
    world_size = args.world_size
    mp.spawn(embed, args=(args.model_path,args.sequence_dir,args.max_length,args.saving_path,args.batch_size,args.world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cleaned_Embed sequences using a pretrained model.")
    # Add an argument for the directory path
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument("sequence_dir", type=str, help="Sequences directory of files to embed file")
    parser.add_argument("max_length", type=int, help="Length to which each sequence will be truncated (for short reads and considering the DNABERT-2 tokenizer for example, it can be as low as 60 tokens)")
    parser.add_argument("saving_path", type=str, help="Where to save the embedded sequences")
    parser.add_argument("batch_size", type=int, default=10000, help="Batch size for embedding")
    parser.add_argument("world_size", type=int, help="Number of processes")
    args = parser.parse_args()
    main(args)


