import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
import torch.nn as nn
from torch.optim import SGD

import random


def fix_randomness(SEED):
    """Fix the randomness for reproducibility

    Args:
        SEED (int): The seed number to fix the randomness to
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cifar10_trainset():
    """Load the trainset of the CIFAR-10 dataset

    Returns:

    """
    data_root_dir = "data"
    # Transform the PILImage images of range[0, 1] to Tensors of normalized range[-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = CIFAR10(
        data_root_dir,
        train=True,
        download=True,
        transform=transform,
    )

    return trainset


def load_cifar10_testset():
    """Load the testset of the CIFAR-10 dataset

    Returns:

    """
    data_root_dir = "data"
    # Transform the PILImage images of range[0, 1] to Tensors of normalized range[-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testset = CIFAR10(
        data_root_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return testset


def generate_dataloader(dataset):
    """Generate the dataloader

    Args:
        dataset (Dataset): The dataset to be loaded into the dataloader

    Returns:

    """
    dataloader = DataLoader(dataset, 50, shuffle=True)

    return dataloader


def show_images(imgs, labels):
    """Visualize 3D RGB Tensor images

    Args:
        imgs (tensor): The 3D RGB Tensor images to visualize
        labels (tensor): The labels of the images

    Returns:

    """
    cifar10_classes = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    if not isinstance(imgs, list):
        imgs = imgs / 2 + 0.5  # Unnormalize
        imgs = [imgs]
        labels = [labels]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(
            title=f"Ground truth: {cifar10_classes[labels[i]]}",
            xticklabels=[],
            yticklabels=[],
            xticks=[],
            yticks=[],
        )


def train_model(model, trainloader, configs):
    """Train a model for image classification

    Args:
        model (nn.Module): The model to train

    Returns:

    """
    device = torch.device(configs["device"])
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=configs["lr"], momentum=configs["momemtum"])

    for epoch in range(configs["num_epochs"]):
        running_loss = 0.0

        for batch_idx, (X_train, y_train) in enumerate(trainloader, 0):
            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()

            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 2000 mini-batches
                print(
                    f"[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

    torch.save(model.state_dict(), f"./model_state_dicts/{configs['save_dir']}")
    return model


def test_model(model, testloader, configs):
    """Test a model for image classification

    Args:
        model (nn.Module): The model to train

    Returns:

    """
    device = torch.device(configs["device"])
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )


def predict_image(model, image):
    """Predict an image with the model and image passed as arguments

    Args:
        model (nn.Module): The model to train
        image (tensor): The image to be predicted
    Returns:

    """
    model.eval()
    with torch.no_grad():
        outputs = model(torch.unsqueeze(image, 0))

        prediction = outputs.squeeze(0).softmax(0)
        predicted_class_id = prediction.argmax().item()
        score = prediction[predicted_class_id].item()

    return predicted_class_id, score
