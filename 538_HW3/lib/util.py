import os
import json

import torch
import torch.nn as nn

from lib.model import DependencyParser


def load_pretrained_model(serialization_dir: str) -> nn.Module:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.pt")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    with open(config_path, "r") as file:
        config = json.load(file)

    # Load Model
    model = DependencyParser(**config)
    model.load_state_dict(torch.load(model_path))

    return model
