# Image Classification Dashboard

A dashboard project that uses the CIFAR-10 image dataset and a model with a simple architecture to classify and predict images.

The CIFAR-10 dataset is an image dataset with ten classes (i.e., airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck).

It has 50,000 training and 10,000 testing data samples. Each image has a (tensor) size of (3, 32, 32) (i.e., RGB image of 32x32 pixels).

The model named "Net" is based on a PyTorch tutorial. It was trained over 50 epochs using the trainset of the CIFAR-10 dataset.

The dashboard is created with PyTorch and Dash.

## Table of Contents

- [Technologies](#technologies)
- [Documentation](#documentation)
- [Setup](#setup)
- [References](#references)
- [Credit](#credit)

## Technologies

- dash
- dash-bootstrap-components
- pandas
- torch
- notebook
- matplotlib
- scipy

## Documentation

The full documentation can be found here (i.e., not out yet).

## Setup

1. Clone the project repository
2. Create a Python virtual environment and install the necessary modules based on the requirement.txt file
3. Create an empty folder named "data"
3. Run the python app.py command

## References

[PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#)

## Credit

[PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#)
