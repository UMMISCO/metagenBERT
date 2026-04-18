# MetagenBERT README

## Introduction

MetagenBERT is a Transformer-based tool for disease prediction from metagenomic data. This model is designed to produce metagenomic samples representations directly from raw shotgun sequencing reads, proposing to describe a metagenome independantly from outside reference databases or taxonomic annotations. This method is adapted to short reads samples.
MetagenBERT creates a representation of a sample by relying on Transformer models such as DNABERT-2 [1] to produce embeddings of DNA reads. Then, it represents the sample by assigning each read to clusters, producing a new representation based on the proportion of reads assigned to each cluster. The method is detailed in [2].

## Pipeline

The metagenome representation is obtained through different steps : 
First, the reads from each samples are embedded by a genomic LLM (originally DNABERT-2 or DNABERT-MS). This step is particularly expensive in computation time and shall be executed using GPUs. To mitigate the cost, experiments have proven that using 5 to 10% of the reads from each sample does not negatively affect the results. This step can be performed with embedding.py script.
On the other hand, a clustering algorithm such as K-Means must be trained in order to produce the representation. It is obtained through clustering of DNA read embeddings. The centroids of the clusters are retrieved and will be used as a partition of the high dimensional embedding space. 
The user can either train his own K-means from his data, or use the MetaCardis [3] model we proposed. Training such a model can be done using the train_kmeans.py script.
After this step, the user can assign the reads from each studied samples to their closest cluster by using the assign_to_kmeans.py script.
Noteworthy, if the user already has K-means centroids, the embed_assign.py allows them to embed the reads and directly assign them to their corresponding cluster while discarding the embedding itself, which can prove very heavy in storage. The authors recommend this method if possible.
The assignments are then cleaned with the purge_clusters.py script, and the proportion of reads assigned to each cluster, or abundance, is obtained with the compute_abundance.py script.
Finally, these representations can be used, in conjunction with others, to train a LASSO classifier with the classification_LASSO.py script.
The theoric pipeline and the computing pipeline can be seen in the figures above.

![MetagenBERT-Glob: The Global Clustering Architecture.Panel (a). For each metagenome, all reads are embedded through the use of a gLLM. A subsample of each embedded metagenome is used to train a K-means common to all the selected dataset. The centroids of the clusters obtained through K-means are retrieved, effectively partitioning the latent space. Panel (b). Every read embedding of each sample is then assigned to its cluster by nearest neighbor search against the centroids, thus creating a new abundance vector based on embeddings rather than species used for classification.](MetagenBERT_pipeline.pdf)

## Dependencies

The following packages are required to use MetagenBERT : 
torch,
numpy,
scikit-learn,
scikit-bio,
faiss-gpu,
transformers,

## Script Usage

In this section, we describe each script, its usage, inputs and outputs

### embedding.py

This script uses a LLM to produce embeddings of all the reads of all samples in a directory and saves them by batch.

usage : 
python embedding.py model_path sequence_dir max_length saving_path batch_size world_size

- model path : the path to the transformer model used to produce the embeddings. Can be a local link or a link to a HuggingFace repository. The authors used DNABERT-2 and provide the local version in this repository (some minor corrections were applied, like disabling FlashAttention for compatibility issues)

- sequence_dir : the path to a directory containing the txt files to embed. The authors strongly advise that one file should correspond to one metagenomic sample. we remind the user that the results show almost none to none performance loss when using only 5-10% of reads per samples, and advise them to directly use files containing only this proportion of reads from the original samples. For computational efficiency, the files should contain only DNA sequences. For example, the 5 first lines of a sample should look like this : 

AGCTAGCTGCTAGATCGATCGATCGATCGATCGTCGATTCGGCTCGATCGGTAGCTAGTAACCCGATCGATGATGAT
GTGCTAGCTTACCGTTTTATTCGCGTATAAAGCATGAGAGATCCTACGATGCCTTTAGCATCACGTACTATATCGAT
GCTTCGTAATGCGCGGCCCGCCATATCGATGCATGCATATAATCTGCACGATCCTCCTGCTCATATATCTACTTCGC
CTAACGTTACGATCCCATCGATGATGCTGGTCGTCGCTGCAATCGACTGCTAGCATGATCTGCAGCTACGAGCTATC
CGTCGTCGCGTGCGTCAACGAAAAAAAAAACGAGCTAGCTAGCTCGATCCGCTACTACGCTGAATAATCGGCCTGAA

- max_length : the length to which each sequence will be truncated or padded before entering the Transformer model. This can depend on the length of the reads to embed and on the tokenizer of the model

- saving_path : the path to the directory where the results will be stored. In this directory, the script will create two new directories named "embeddings" and "idx". In each of these two directories, one directory per sample will be created, in which the outputs will effectively be stored.
Considering the sample named S1, in /saving_path/embeddings/S1/ , the program will save one embedding file per batch under the name embeddings_X_Y.pt with X the rank of the batch and Y the rank of the GPU. Therefore, if S1 contains 100.000 reads and is embedded on 4 GPUs with a batch size of 10.000, the directory will contain 12 files named embeddings_{0,1,2}_{0,1,2,3}.pt, with embeddings_{0,1}_{0,1,2,3}.pt containins 10.000 embeddings each and embeddings_{2}_{0,1,2,3}.pt containing 5.000 embeddings each, totalling 100.000.
The "idx" directory follows the same structure, with files named "idx_X_Y.pt". These files simply keep track of the line number in the original sample file corresponding to each embedding, allowing the user to associate each embedding to the sequence it originates from.

- batch_size : the number of sequences treated at once by each GPU. From our experience, using DNABERT-2, A100 GPUs could be used with a batch size of 35.000 - 40.000 sequences

- world_size : the number of processes (and therefore GPUs) the program will have access to. This script was only tested on a single node. The authors adapted a version usable on French supercalculator Jean-Zay [4] in a multi-node configuration. This requires the use of the package idr_torch and is not adapted to most configurations. This script can be demanded to the authors.

### train_kmeans.py

This script uses a dataset of embedding directories as obtained from the usage of embedding.py and loads a small number of reads from each sample. It then uses FAISS [5] package to perform K-Means clustering with GPU acceleration.

usage:
python train_kmeans.py data_path save_path n_clusters_list n_iter verbose min_points max_points n_to_load

- data_path : the path to the directory of saved embeddings as obtained through embeddings.py. This directory shall contain one subdirectory per sample the user wishes to use for training the K-means model. Each subdirectory shall contain the embeddings_X_Y.pt files obtained before. The authors advise the user not to merge the files obtained before, as the resulting file would probably be very heavy and could lead to issues in RAM. No other file than the embeddings_X_Y.pt files should be in these directories

- n_clusters_list : to allow the user to perform training with various numbers of clusters and compare the results in one step without needing to load the data anew, this parameter takes as input a string that will be parsed to a list. Please write the list exactly under the form [16,32,64] without spaces or additional characters. You can use a list of one element.

- save_path : the path to the directory where both centroids and faiss index will be saved under the names centroids_NN.npy and index_NN.npy, NN being the number of clusters in the index to be saved. Only the centroids file will be used by MetagenBERT for the rest of the pipeline

- n_iter : number of iterations for the K-means algorithm 

- n_to_load : number of reads to use for clustering per sample

- verbose : activate FAISS verbose mode (progress and stats during clustering)

- min_points : minimum number of points to use for clustering. This number will be multiplied by the number of clusters. The authors recommand to set it low

- max_points : maximum number of points to use for clustering. This number will be multiplied by the number of clusters. The authors recommand to set it high enough so that max_points*number_of_clusters > n_to_load*number_of_files

### assign_to_kmeans.py

This script uses a dataset of embedding directories as obtained from the usage of embedding.py and loads all their reads by batch, then assigns them to centroids using FAISS [4] package with GPU acceleration.

usage :
python assign_to_kmeans.py save_path data_path idx_path centroid_path_list nb_batch

- save_path : path to save the clustering_results. This directory will contain one directory per path to centroid file given, in which will be created one directory per sample, which will each contain various files of types assignments_N.npy and idx_N.npy with N the batch id. assignments contain the centroids to which the read of corresponding id was assigned

- data_path : the path to the directory of saved embeddings as obtained through embeddings.py. This directory shall contain one subdirectory per sample the user wishes to use for training the K-means model. Each subdirectory shall contain the embeddings_X_Y.pt files obtained before. The authors advise the user not to merge the files obtained before, as the resulting file would probably be very heavy and could lead to issues in RAM. No other file than the embeddings_X_Y.pt files should be in these directories.

- idx_path : the path to the directory of saved indexes as obtained through embeddings.py. This directory shall contain one subdirectory per sample the user wishes to use for training the K-means model. Each subdirectory shall contain the idx_X_Y.pt files obtained before. The authors advise the user not to merge the files obtained before, in order not to break the correspondance with embeddings. No other file than the idx_X_Y.pt files should be in these directories.

- centroid_path_list : to allow the user to perform assignment with various centroids files and compare the results in one step without needing to load the data anew, this parameter takes as input a string that will be parsed to a list. Please write the list of paths to centroids npy files exactly under the form ["/absolute/path/to/centr16/centroids_16.npy","/absolute/path/to/centr32/centroids_32.npy","/absolute/path/to/centr64/centroids_64.npy"] without spaces or additional characters. You can use a list of one element. Keep in mind the name of your centroid file stripped of ".npy" will be used as directory name to save the results. You can use any numpy array of dimensions (number_of_centroids, number_of_features). Make sure the number of features correspond to the number of features of your embeddings.

- nb_batch : number of embeddings_X_Y.pt files to load in an assignments batch, depending on your RAM

### embed_assign.py

This script can be used if you already have centroids to directly embed reads and assign them without needing to save them in between. It therefore replaces embedding.py and assign_to_kmeans.py and considers you have centroids similar to the output of train_kmeans.py

usage :
python embed_assign.py model_path sequence_dir max_length saving_path batch_size world_size centroids_path

First six arguments are equivalent to those from embedding.py. Last one is equivalent to centroid_path_list from assign_to_kmeans.py

- model path : the path to the transformer model used to produce the embeddings. Can be a local link or a link to a HuggingFace repository. The authors used DNABERT-2 and provide the local version in this repository (some minor corrections were applied, like disabling FlashAttention for compatibility issues)

- sequence_dir : the path to a directory containing the txt files to embed. The authors strongly advise that one file should correspond to one metagenomic sample. we remind the user that the results show almost none to none performance loss when using only 5-10% of reads per samples, and advise them to directly use files containing only this proportion of reads from the original samples. For computational efficiency, the files should contain only DNA sequences. For example, the 5 first lines of a sample should look like this : 

AGCTAGCTGCTAGATCGATCGATCGATCGATCGTCGATTCGGCTCGATCGGTAGCTAGTAACCCGATCGATGATGAT
GTGCTAGCTTACCGTTTTATTCGCGTATAAAGCATGAGAGATCCTACGATGCCTTTAGCATCACGTACTATATCGAT
GCTTCGTAATGCGCGGCCCGCCATATCGATGCATGCATATAATCTGCACGATCCTCCTGCTCATATATCTACTTCGC
CTAACGTTACGATCCCATCGATGATGCTGGTCGTCGCTGCAATCGACTGCTAGCATGATCTGCAGCTACGAGCTATC
CGTCGTCGCGTGCGTCAACGAAAAAAAAAACGAGCTAGCTAGCTCGATCCGCTACTACGCTGAATAATCGGCCTGAA

- max_length : the length to which each sequence will be truncated or padded before entering the Transformer model. This can depend on the length of the reads to embed and on the tokenizer of the model

- saving_path : the path to the directory where the results will be stored. In this directory, the script will create two new directories named "embeddings" and "idx". In each of these two directories, one directory per sample will be created, in which the outputs will effectively be stored.
Considering the sample named S1, in /saving_path/embeddings/S1/ , the program will save one embedding file per batch under the name embeddings_X_Y.pt with X the rank of the batch and Y the rank of the GPU. Therefore, if S1 contains 100.000 reads and is embedded on 4 GPUs with a batch size of 10.000, the directory will contain 12 files named embeddings_{0,1,2}_{0,1,2,3}.pt, with embeddings_{0,1}_{0,1,2,3}.pt containins 10.000 embeddings each and embeddings_{2}_{0,1,2,3}.pt containing 5.000 embeddings each, totalling 100.000.
The "idx" directory follows the same structure, with files named "idx_X_Y.pt". These files simply keep track of the line number in the original sample file corresponding to each embedding, allowing the user to associate each embedding to the sequence it originates from.

- batch_size : the number of sequences treated at once by each GPU. From our experience, using DNABERT-2, A100 GPUs could be used with a batch size of 35.000 - 40.000 sequences

- world_size : the number of processes (and therefore GPUs) the program will have access to. This script was only tested on a single node. The authors adapted a version usable on French supercalculator Jean-Zay [5] in a multi-node configuration. This requires the use of the package idr_torch and is not adapted to most configurations. This script can be demanded to the authors

- centroids_paths : to allow the user to perform assignment with various centroids files and compare the results in one step without needing to load the data anew, this parameter takes as input a string that will be parsed to a list. Please write the list of paths to centroids npy files exactly under the form ["/absolute/path/to/centr16/centroids_16.npy","/absolute/path/to/centr32/centroids_32.npy","/absolute/path/to/centr64/centroids_64.npy"] without spaces or additional characters. You can use a list of one element. Keep in mind the name of your centroid file stripped of ".npy" will be used as directory name to save the results. You can use any numpy array of dimensions (number_of_centroids, number_of_features). Make sure the number of features correspond to the number of features of your embeddings.

### purge_clusters.py

Merges all assignments and index files from each sample in a directory into single files. Also removes a few doubles that can appear due to batching in embeddings (maximum a few hundreds, negligible)

usage : 
python purge_clusters.py dataset_path --num_workers XX

- dataset_path : path to the directory of the dataset. This directory should be similar to the output of either assign_to_kmeans.py or embed_assign.py, i.e. contain a directory per sample, which contains assignments_NN.py and idx_NN.py files. It will then create in these directories two files : assignments.npy and indexes.npy, which are merges of all former files. Caution : assignments_NN.npy and idx_NN.npy files will be deleted after this step

- --num_workers : number of CPUs to process in parallel. Default is set to the number of CPUs available

### compute_abundance.py

Computes the proportion of reads assigned to each cluster for a whole dataset

usage :
python compute_abundance.py dataset_path number --num_workers XX

- dataset_path : path to the directory of the dataset. This directory should be similar to the output of purge_clusters.py, i.e. contain a directory per sample, which contains one file named assignments.npy and one file named indexes.npy, of same length. It will then create in these directories one numpy file of dimension (number of clusters) named abundance.npy, which contains the proportion of reads assigned to each cluster

- number : the number of clusters used for assignment

- --num_workers : number of CPUs to process in parallel. Default is set to the number of CPUs available

### classification_LASSO.py

trains a LASSO model to classify a given dataset and saves it for further use. Also accepts complementary sources of data such as taxonomic abundance

usage :
python classification_LASSO.py --metagenbert_path  /path/to/dataset --corresp /path/to/labels --additional_paths "["/path/to/dataset1","/path/to/dataset2"]" --ilr_transform boolean

- --metagenbert_path : path to the directory of the dataset. This directory should be similar to the one output by compute_abundance.py, i.e. contain one directory per sample, which contain an abundance.npy file.

- --corresp : path to the labels file. File should be json containing one directory where keys are sample ids (identical to the names of directories in the dataset, where abundance.npy files are stored) and values are int labels

- --additional_paths : paths to files containing additional compositional information to add, for example to compare performances with MetagenBERT embedding and the combination of MetagenBERT and species based embeddings. To allow the user to add several sources of information, this parameter takes as input a string that will be parsed to a list. Please write the list of paths to data exactly under the form ["/absolute/path/to/data1.csv","/absolute/path/to/data2.csv"] without spaces or additional characters. You can use a list of one element. Data should be in csv of 2 columns with sample ids as left element and abundance (between 0 and 1, not percentage) as right element.

- --ilr_transform : whether to apply ilr transform to each source of data independantly

[1] Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana V Davuluri, and Han Liu. DNABERT-2: Efficient foundation model and benchmark for multi-species genomes. The Twelfth International Conference on Learning Representations, 2024.

[2] Gaspar Roy, Eugeni Belda, Baptiste Hennecart, Yann Chevaleyre, Edi Prifti, Jean-Daniel Zucker. MetagenBERT: a Transformer-based Architecture using Foundational genomic Large Language Models for novel Metagenome Representation. arXiv. 2026.

[3] S.K Forslund, R. Chakaroun, M. Zimmermann-Kogadeeva, et al}. Combinatorial, additive and dose-dependent drug–microbiome associations}. Nature}. 2021. 10.1038/s41586-021-04177-9.

[4] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, Hervé Jégou. The Faiss library. arXiv. 2024.

[5] http://www.idris.fr/jean-zay/jean-zay-presentation.html
