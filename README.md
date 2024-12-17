# EC523-Deep-Learning-Project
## 2D TO 3D RECONSTRUCTION OF MEDICAL IMAGES USING UNSUPERVISED GANs
### Project Description
The goal of this project was to make use of unsupervised/weakly supervised learning to reconstruct 3D images from biplanar 2D chest XRay inputs. Various architectures were experimented with and tested. 
Please refer to our report for more information on the project. Each of the folders correspond to the different architectures and experiments that were carried out.

## Team Members
- Adwait Kulkarni
- Avantika Kothandaraman
- Harshvardhan Shukla
- Shivam Goyal

## Folders in the repository
**uX2CTGAN** - Contains the .ipynb file with code demonstrations and experiments for the uX2CTGAN architecture.

**CycleGAN-B** - Contains all the .py files needed to execute the CycleGAN-B architecture that takes in biplanar XRays as input - along with the .ipynb implemented code.

**CycleGAN-F** - Contains all the .py files needed to execute the CycleGAN-B architecture that takes in frontal XRays as input - along with the .ipynb implemented code.

**3DCNN** - Contains the .ipynb files that demonstrate the experiments with 3DCNNs

**Diffusion Model** - Contains .ipynb files and some important outputs obtained with the diffusion model

**Misc** - Contains all miscellaneous code files that were used during this project.

## How to Run the Project

### Prepare the environment
We recommend using an interactive environment like Jupyter Notebooks. The configuration of the code in our notebooks explains how to load the data, preprocess it and use it to train the models. It must be ensured that all the necessary packages are installed like so:

```bash
pip install numpy matplotlib torch torchvision scipy scikit-learn tqdm
```

### Download the dataset
The dataset and its corresponding .csv file with the descriptive data of the images can be downloaded from this link: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university. </br> Once downloaded, the data and the .csv file must be stored in the same directory as the .ipynb notebooks that you are currently working with. 

### Loading the dataset
An essential part of our data handling framework, this subclass of `torch.utils.data.Dataset` automates the loading, processing, and pairing of image data from the specified directory. It handles loading the image, preprocessing it, removing data files that have missing counterparts (i.e., if some frontal images are missing their corresponding lateral images and vice-versa) and pairs each image_id with their projection type (frontal/lateral). This data loading is different in each of our architectures but follow essentially the same process. 

### Running the code
Once the data is loaded, the same workflow or order of steps can be followed for each of the architectures as displayed in the Jupyter Notebooks of each folder (of each architecture).




