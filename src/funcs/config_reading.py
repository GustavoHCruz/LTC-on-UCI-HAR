import json
from typing import List


def read_configs():
  """
  Reads the configurations JSON file, validates the keys and their types, and returns the retrieved values.

  Args:

  Returns:
    dict: A dictionary with the retrieved values from the JSON.

  Raises:
    ValueError: If a key is missing in the JSON.
    TypeError: If a value's type is incorrect.
  """
  # Define the expected types
  expected_types = {
    "model_name": str,
    "processing": {
      "save_path": str,
      "validation_proportion": float,
    },
    "training": {
      "seed": int,
      "float_precision": str,
      "batch_sizes": List[int or "all"],
      "num_neurons": int,
      "learning_rate": float,
      "max_epochs": int
    }
  }

  with open("config.json", 'r') as file:
    data = json.load(file)

  for key, expected_type in expected_types.items():
    if key not in data:
      raise ValueError(f"Missing key: {key}")
    
    # Special validation for batch_sizes to ensure it's a list of exactly 3 integers
    if key == "batch_sizes":
      if not isinstance(data[key], list) or not all(isinstance(i, int) for i in data[key]) or len(data[key]) != 3:
        raise TypeError(f"Incorrect type or size for {key}: Expected a list of 3 integers")
    else:
      if not isinstance(data[key], expected_type):
        raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
    
    return {
    "model_name": data["model_name"],
    "processing": {
      "save_path": data["processing"]["save_path"],
      "validation_proportion": data["processing"]["validation_proportion"],
    },
    "training": {
      "seed": data["training"]["seed"],
      "float_precision": data["training"]["float_precision"],
      "batch_sizes": data["training"]["batch_sizes"],
      "num_neurons": data["training"]["num_neurons"],
      "learning_rate": data["training"]["learning_rate"],
      "max_epochs": data["training"]["max_epochs"],
    },
  }
