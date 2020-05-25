# Introduction to Machine Learning Nanodegree
## Deep Learning
### Create your own Image Classifier

#### Overview
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

#### Install
This project requires Python 2.7 and the following Python libraries installed:
- PyTorch
- ArgParse
- Jason
- PIL
- NumPy
- Pandas
- matplotlib
- scikit-learn You will also need to have software installed to run and execute an iPython Notebook
We recommend students install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

#### Run
In a terminal or command window, navigate to the top-level project directory / (that contains this README) and run one of the following commands:

ipython notebook Image Classifier Project.ipynb or

jupyter notebook Image Classifier Project.ipynb This will open the iPython Notebook software and project file in your browser.

**Or for Command Line**
In a terminal or command window, navigate to the top-level project directory / (that contains this README) and run one of the following commands:
- Train a new network on a data set with train.py

  - Basic usage: python train.py data_directory
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg13"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    - Use GPU for training: python train.py data_dir --gpu
- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

  - Basic usage: python predict.py /path/to/image checkpoint
  - Options:
    - Return top KK most likely classes: python predict.py input checkpoint --top_k 3
    - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    - Use GPU for inference: python predict.py input checkpoint --gpu
