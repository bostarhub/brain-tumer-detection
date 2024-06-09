
# Brain Tumor Detection

This repository contains a project aimed at detecting brain tumors from MRI images using Convolutional Neural Networks (CNN) with TensorFlow. The model is trained and saved as `brain_tumor_detection.h5`.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Brain tumor detection is a crucial task in medical imaging. This project uses deep learning techniques to classify MRI images into two categories: `tumor` and `no tumor`. The goal is to assist medical professionals in diagnosing brain tumors with higher accuracy and efficiency.

## Dataset
The dataset used in this project is sourced from Kaggle, containing 300 MRI images. The dataset is split into training and validation sets to ensure the model is trained effectively and its performance is validated. Below is a sample image from the dataset:

## Installation
To get started with the project, follow these steps to clone the repository and install the required packages:

git clone https://github.com/ishratjahan06/brain-tumor-detection.git
cd brain-tumor-detection
pip install -r requirements.txt


### Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Model Architecture
The CNN model used for this project consists of several convolutional layers followed by max-pooling layers and dense layers. This architecture is designed to extract features from the images and make accurate classifications. Below is a diagram of the model architecture:

## Training
The model is trained using the preprocessed MRI images. The training process involves feeding the images into the CNN model, which learns to distinguish between images with and without tumors. The model is then validated on a separate set of images to ensure its accuracy. After training, the model is saved as `brain_tumor_detection.h5`.

## Results
The model achieves an accuracy of approximately 95% on the validation set. The training process is monitored using accuracy and loss metrics, and the results are visualized in the graph below:

## Usage
To use the trained model for predictions, load the model and pass the MRI images for prediction. The model will classify each image as either containing a tumor or not. This can assist medical professionals in making diagnostic decisions.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please open an issue or submit a pull request. We appreciate your help in improving the project.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.

This layout presents each section within a box, improving readability and organization in the `README.md` file.
