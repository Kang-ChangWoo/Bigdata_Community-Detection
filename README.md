# Dual Neural Network for Community Detection

MICA is a clustering tool for single-cell RNA-seq data. MICA takes a preprocessed gene expression matrix as input and
efficiently cluster the cells.
MICA consists of the following main components:
1. Mutual information estimation for cell-cell distance quantification
2. Dimension reduction on the non-linear mutual information-based distance space
3. Consensus clustering on dimension-reduced spaces
4. Clustering visualization and cell type annotation

MICA workflow:

<img src="images/MICA_workflow.png" width="500">


## Prerequisites
* [python>=3.7.6, <=3.9.2](https://www.python.org/downloads/) (developed and tested on python 3.7.6, tested on python3.9.2)
    * See [requirements.txt](https://github.com/jyyulab/MICA/blob/million/requirements.txt) file for other dependencies


## Installation
#### Using conda to create a virtual environment 
##### (Not available until this line is removed)
The recommended method of setting up the required Python environment and dependencies 
is to use the [conda](https://conda.io/docs/) dependency manager:
```
conda create -n mica100 python=3.9.2        # Create a python virtual environment
source activate mica100                     # Activate the virtual environment
pip install MICA                            # Install MICA and its dependencies
```

#### Install from source
```
conda create -n mica100 python=3.9.2        # Create a python virtual environment
source activate mica100                     # Activate the virtual environment
git clone https://github.com/jyyulab/MICA   # Clone the repo
cd MICA                                     # Switch to the MICA root directory
pip install .                               # Install MICA from source
mica -h                                     # Check if mica works correctly
```


## 1 Reorganizing file/folder structure.
We use 4 dataset as below:
 - `Real-world dataset` (for test)
 - `Synthesized real-world-like dataset` (for train)
 - `TC1 dataset` (for test)
 - `Synthesized TC1-like dataset` (for train)

Each file and folder structure is unique.

For example, the `real-world dataset` and the `TC1 dataset` are saved using a nested directory structure. 
In contrast, the `synthesized real-world-like dataset` and the `synthesized TC1-like dataset` use a flat directory structure.

Additionally, the `real-world dataset` files are named with descriptive text, such as "dolphin," "football," and "karate." 
On the other hand, the `TC1 dataset` files are named using sequential indexing.

Therefore, we have to re-organize given dataset structures.

For `real-world dataset`, we type below:
```
python N1-organize_dataset.py                      \          # 
  --dataset_path 'path/to/your/real-world_dataset' \          # 
  --isit_hierarchy 1                               \          #
  --testset real                                              #
```

For `TC1 dataset`, we type below: 
```
python N1-organize_dataset.py                      \          # 
  --dataset_path 'path/to/your/TC1-dataset'        \          # 
  --isit_hierarchy 1                               \          #
  --testset TC1                                               #
```


## 2 Precomputing *peak resolution parameters.
\* peak means best performed resolution parameter according to each graph data.







## 3 Training proposed network.
we mainly train on synthesized dataset, and then we test/eval on given dataset.

For `real-world dataset`, we type below:
```
python N3-train.py                                                  \          # 
  --train_path 'path/to/your/`synthesized real-world-like dataset`  \          # 
  --test_path 'path/to/your/`real-world_dataset`                    \          #
  --testset real                                                    \          #
```

For `TC1 dataset`, we type below: 
```
python N3-train.py                                                  \          # 
  --train_path 'path/to/your/`synthesized TC1-like dataset`         \          # 
  --test_path 'path/to/your/`TC1 dataset`                           \          #
  --testset TC1                                                     \          #
```



## 5 Visualizing results
We utilze `N5-visualize-result.py` file to get proper files(.xlsx/.csv) and manually edit results in editing tools such as Microsoft Powerpoint and Excel.
Please see the submitted paper to detailed visualization.
