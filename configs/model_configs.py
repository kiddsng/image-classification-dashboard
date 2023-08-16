import torch


def get_model_configs(model_name):
    """Return the model's configurations class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Model not found: {}".format(model_name))
    return globals()[model_name]


class Net:
    def __init__(self):
        super(Net, self)

        self.name = "Net"
        self.model_state_dict = torch.load("./model_state_dicts/cifar_net.pth")
