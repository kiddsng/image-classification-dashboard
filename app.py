from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

import torch
import torchvision.transforms.functional as F
import numpy as np

from configs.model_configs import get_model_configs
from models.models import get_model_class
from utils import (
    load_cifar10_testset,
    generate_dataloader,
    predict_image,
)

# Initialize the models and configurations used in the Dash app
# Change the values below
app_data = {
    "cifar10_classes": (
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
    ),
}

# Load the datasets, dataloaders, and models
print("Generating and loading dataset...")
cifar10_testset = load_cifar10_testset()
print("Done.")
print("Generating and loading dataloader...")
cifar10_testloader = generate_dataloader(cifar10_testset)
print("Done.")
print("The app is ready to use.")

# Initialize the Dash app with a Dash Bootstrap Lumen theme
external_stylesheets = [dbc.themes.LUMEN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "3D RGB Image Classification Dashboard",
                                    className="fw-bold text-primary text-uppercase",
                                ),
                                html.P(
                                    "Image Classification for the CIFAR-10 Dataset",
                                    className="fs-5 text-secondary",
                                ),
                            ],
                            className="pt-5 pb-2",
                        ),
                    ],
                ),
            ],
        ),
        # Main
        dbc.Row(
            [
                # Settings
                dbc.Col(
                    [
                        dcc.Store(id="data-storage"),
                        # Configuration settings
                        html.Div(
                            [
                                html.H2(
                                    "Dashboard Settings",
                                    className="fs-5 fw-bold text-uppercase mt-2 mb-3",
                                    style={"letterSpacing": "1.25px"},
                                ),
                                html.P(
                                    "The images are classified with the Net model. It is based on a PyTorch tutorial and has a simple network architecture.",
                                    className="text-break mt-2",
                                ),
                                html.P(
                                    "It is trained using the trainset of the CIFAR-10 dataset. The trainset consists of 50,000 image data samples, and 10 classes (i.e., airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck). The model is trained over 50 epochs.",
                                    className="text-muted mt-2",
                                ),
                            ],
                            className="bg-light rounded-3 py-3 px-4 mb-3",
                        ),
                    ],
                    width=12,
                    lg=4,
                ),
                # Image visualization and classification
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="data-visualization", className="mb-3"),
                                dbc.Button(
                                    "Generate",
                                    color="success",
                                    id="generate-button",
                                    className="me-3 mb-3",
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Ground truth: ",
                                                html.Span(
                                                    id="ground-truth-output",
                                                    className="fw-bold",
                                                ),
                                            ],
                                            className="bg-light rounded-3 text-center py-3 px-4",
                                        ),
                                        html.P(
                                            [
                                                "Predicted class: ",
                                                html.Span(
                                                    id="predicted-class-output",
                                                    className="fw-bold",
                                                ),
                                            ],
                                            className="bg-light rounded-3 text-center py-3 px-4",
                                        ),
                                    ],
                                    className="d-flex flex-row gap-2",
                                ),
                            ],
                            className="d-flex flex-column align-items-center",
                        ),
                    ],
                    width=12,
                    lg=8,
                ),
            ],
            className="gy-3",
        ),
    ],
)


# Generate a random image from the selected dataset
@app.callback(
    Output("data-visualization", "figure"),
    Output("ground-truth-output", "children"),
    Output("predicted-class-output", "children"),
    Output("data-storage", "data"),
    Input("generate-button", "n_clicks"),
)
def generate(n):
    # Generate a random image when the app starts or when the user clicks on the generate button
    cifar10_dataiter = enumerate(cifar10_testloader)
    batchidx, (images, labels) = next(cifar10_dataiter)  # "next" randomly
    random_image_data, random_image_label = (
        images[0],
        labels[0],
    )  # index randomly, instead of 0
    random_image = F.to_pil_image(random_image_data / 2 + 0.5)
    fig = px.imshow(
        np.asarray(random_image),
    )
    fig.update_layout(width=320, height=320)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    selected_model = get_model_class("Net")()
    model_configs = get_model_configs("Net")()
    selected_model.load_state_dict(model_configs.model_state_dict)

    predicted_class_id, score = predict_image(selected_model, random_image_data)
    predicted_class = app_data["cifar10_classes"][predicted_class_id]
    prediction = f"{predicted_class} ({score * 100:.1f}%)"

    image_label = app_data["cifar10_classes"][random_image_label]
    image_data = {
        "image_data": random_image_data,
        "image_label": image_label,
    }

    return fig, image_label, prediction, image_data


if __name__ == "__main__":
    app.run_server(debug=True)
