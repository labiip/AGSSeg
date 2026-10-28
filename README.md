# AGSSeg
* This project is the implementation for AGSSeg.

# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).

# Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python >= 3.6 
* Efficientnet-Pytorch `pip install efficientnet_pytorch`
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone this project
```
git clone https://github.com/labiip/AGSSeg
cd AGSSeg
```
2. Train the model
```
cd code
python train.py
```
3. Test the model
```
python test.py
```

# Acknowledgement
* The codebase is adapted from the work [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

