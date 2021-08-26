 # Renal Classification
This project aim to assess the benifit of multi-task learning in classification of renal lesion based on MRI images and clinical data. There is three task on which our models will be trained: the classification of the malignancy, the subtype and the grade of renal tumour. The classification of the subtype and the grade can only be done on malignant tumour.
 
 ## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Task List](#task-list)
* [Folder Structure](#folder-structure)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)

## General info
Please take note that the dataset has not been push on github since it include private data and it is really heavy.
For any information about the data that has been used, please contact Alexandre Ayotte (Alexandre.Ayotte2@USherbrooke.ca) or Martin Vallières (Martin.Vallieres@USherbrooke.ca)

## Requirements
The present package is written in **Python 3.8**. In order to run it in full capacity, the user should have a **Nvidia GPU** with **CUDA 11.1** installed.
Morever, the following package are required to execute our code.

- antspy
- h5py
- matplotlib
- monai
- nibabel
- pandas
- pytorch
- scikit-learn
- torchsummary
- tqdm
- comet_ml
- tensorboard


## Task List
- [x] Normalization
  - [x] MRIimage Class
  - [x] Patient Class
  - [x] Transfer header from images to ROI
  - [x] Normalize and crop 3D images and ROI
  - [x] Normalize and crop 2D images and ROI
- [x] Classification 
  - [x] RenalDataset
  - [x] Abstract Trainer
  - [x] NeuralNet
- [x] Single-Task Learning
  - [x] ResNet2D
  - [x] ResNet3D
  - [x] SingleTaskTrainer
  - [x] Add tensorboard
- [x] Prepare Multi-Task Learning
  - [x] MultiTaskTrainer
  - [x] MultiLevelResNet3D
  - [x] SharedNet
  - [x] Uncertainty loss
  - [x] Conditional probability
  - [x] Cumulate gradient
  - [x] Add hparam tracking with comet.ml
  - [x] Add Hybrid ResNet
  - [x] Add MTAN
  - [x] Compute Radiomics
- [x] Experiment Phase 1
  - [x] Block type
  - [x] Hybrid ResNet
  - [x] Grouped Convolution
  - [x] Merged Mask
  - [x] Saving criterion
- [x] Experiment Phase 2
  - [x] MTL Loss (H-S)
  - [x] Split layer (H-S)
  - [x] Pretraining (S-S)
  - [x] Sharing Modules Position (S-S)
  - [x] MTAN depth (MTAN)
  - [x] MTAN width (MTAN)
  - [x] Attention Module (MTAN)
- [ ] Prepare Radiomics Experiments
  - [x] Compute radiomics
  - [ ] Adapt MultiTaskTrainer
  - [x] Add Learn-To-Branch (LTB) Model
- [ ] Experiment Phase 3
  - [ ] Auxiliary tasks
  - [ ] Base Test (LTB)
  - [ ] Multiple Layer Type (LTB)
- [ ] Experiment Phase 4
  - [ ] Radiomics in self-supervising
    - [ ] With one set of radiomics
    - [ ] All sets of radiomics
  - [ ] Radiomics in MultiTask Learning
    - [ ] One task + ine radiomics set
    - [ ] All tasks + one radiomics set
    - [ ] One task + all radiomics sets
    - [ ] All tasks + all radiomics sets
- [ ] Write paper
- [ ] Bonus
  - [ ] Add Fully Adaptive Features Sharing (FAFS) Model
  - [ ] Test FAFS

## Folder Structure
 ```
.
├── Classification
│   ├── ArgParser.py                # ArgParser used by all Main files.
│   │── comet_api_key.txt           # Contain the needed api key to use comet.ml
│   ├── Constant.py                 # Several class of constant and enum.
│   │
│   ├── Data_manager                # Dataset and data visualisation related files.
│   │   ├── DataAugView2D.py        # Visualisation of the data augmentation on 2D images.
│   │   ├── DataAugView3D.py        # Visualisation of the data augmentation on 3D images.
│   │   ├── DatasetBuilder.py       # Build and split the training, validation and testing set.
│   │   └── RenalDataset.py         # RenalDataset class.
│   │
│   ├── LTBMain.py                  # Main script for experimentation on the LTBResNet.
│   │
│   ├── Model                       # Neural network model related files.
│   │   ├── Block.py                # Commun ResNet, PreResNet and Attention block.
│   │   ├── CapsNet2D.py            # CapsNet2D class.
│   │   ├── CapsuleBlock.py         # CapsNet related block and function.
│   │   ├── HardSharedResNet.py     # HardSharing ResNet3D class.
│   │   ├── LTBResNet.py            # Learn-To-Branch ResNet3D class.
│   │   ├── Module.py               # Commun module used to build NeuralNetwork.
│   │   ├── MTAN.py                 # The MTAN class (MultiTask Attention Network).
│   │   ├── NeuralNet.py            # Abstract class of all neural network classes (except ResNet2D).
│   │   ├── ResNet_2D.py            # ResNet2D model class.
│   │   ├── ResNet.py               # ResNet3D class.
│   │   └── SharedNet.py            # SharedNet model class.
│   │
│   ├── MtanMain.py                 # Main script for experimentation on the Mtan.
│   ├── MultiTaskMain.py            # Main script for experimentation on the Hard-Sharing ResNet3D.
│   ├── SharedNetMain.py            # Main script for experimentation on the SharedNet.
│   ├── STL2DMain.py                # Main script for experimentation on the ResNet2D.
│   ├── STL3DMain.py                # Main script for experimentation on the ResNet3D.
│   │
│   ├── Trainer                     # Neural Network training related files.
│   │   ├── MultiTaskTrainer.py     # Training class for Multi-Task Learning.
│   │   ├── SingleTaskTrainer.py    # Training class for Single-Task Learning.
│   │   ├── Trainer.py              # Abstract class of all trainer classes.
│   │   └── Utils.py                # Utils function for the trainers.
│   │
│   └── Utils.py                    # Utils function for the main files.
│
├── Normalization                   # Normalization and preprocessing of MRI images.
│   ├── MRI_image.py                # The MRIimage class.
│   ├── Normalize2D.py              # Used to create a h5py dataset of 2D normalized images.
│   ├── Normalize.py                # Used to normalize the nifti images of a list of patient.
│   ├── Patient.py                  # The Patient class.
│   ├── RoiStats.py                 # Script used to compute statistics about the ROI.
│   ├── TransferHeader.py           # Script used to transfer metadata between nifti files.
│   ├── Transfer_in_hdf5.py         # Transfer nifti files and clinical data into hdf5.
│   ├── Utils.py                    # Commun and useful function in normalization process.
│   └── VisualizeNormalization.py   # Script used to visualize the normalization process.
│
└── README.md
 ```

## Authors
- Alexandre Ayotte (Ms student in medical imaging, University of Sherbrooke, GRIIS)
- Martin Vallieres (Pr, Computer science departement, University of Sherbrooke, GRIIS) 

## Acknowledgments
Merci à Simon Giard-Leroux pour avoir fait mes impôts de 2019 et 2020.
