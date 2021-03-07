 # Renal Classification

 
 ## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Folder Structure](#folder-structure)

## General info


## Requirements

The present package is written in **Python 3.8**. In order to run it in full capacity, the user should have a **Nvidia GPU** with **CUDA 11.1** installed.
Morever, the following package are required to execute our code.
```
-AntsPy
-h5py
-MatPlotLib
-MONAI
-nibabel
-Pytorch
-Scikit-Learn
-torchsummary
-tqdm
```

## Folder Structure
 ``` bash
.
├── Classification                         
│   ├── CrossStitchMain.py             # The main script for the experimentation on the SharedNet
│   │
│   ├── Data_manager                   # Dataset and data visualisation related files
│   │   ├── DataAugView2D.py           # Visualisation of the data augmentation on 2D images
│   │   ├── DataAugView3D.py           # Visualisation of the data augmentation on 3D images
│   │   ├── DataManager.py             # RenalDataset class and split trainset function
│   │
│   ├── Main_ResNet2D.py               # The main script for the experimentation on the ResNet2D
│   │
│   ├── Model                          # Neural network model related files
│   │   ├── Module.py                  # Commun module used to build NeuralNetwork
│   │   ├── NeuralNet.py               # Abstract class of all neural network classes (except ResNet2D)
│   │   ├── ResNet_2D.py               # ResNet2D model class
│   │   ├── ResNet.py                  # ResNet block, ResNet3D and multi level ResNet3D classes
│   │   └── SharedNet.py               # SharedNet model class
│   │
│   ├── MultiTaskMain.py               # Main experimentation on the multi level ResNet3D
│   ├── SingleTaskMain.py              # Main experimentation on the ResNet3D
│   │
│   └── Trainer                        # Neural Network training related files
│       ├── MultiTaskTrainer.py        # Training class for Multi-Task Learning
│       ├── SingleTaskTrainer.py       # Training class for Single-Task Learning
│       ├── Trainer.py                 # Abstract class of all trainer classes
│       └── Utils.py                   # Utils function for the trainers
│
├── Normalization                      # Normalization and preprocessing of MRI images
│   ├── MRI_image.py                   # MRIimage class used to represent an MRI image and it ROI
│   ├── Normalize2D.py                 # Script used to create a h5py dataset of 2D normalized images
│   ├── Normalize.py                   # Script used to normalize the nifti images of a list of patient
│   ├── Patient.py                     # The Patient class used to represent the 2 MRIimage of a patient
│   ├── RoiStats.py                    # Script example that show how to compute statistics about the ROI
│   ├── TransferHeader.py              # Script used to transfer metadata between nifti files
│   ├── Transfer_in_hdf5.py            # Used to transfer nifti files and clinical data into hdf5 dataset
│   ├── Utils.py                       # Commun and useful function used by patient and MRIimage classes
│   └── VisualizeNormalization.py      # Script used to visualize the normalization process
│
└── README.md
