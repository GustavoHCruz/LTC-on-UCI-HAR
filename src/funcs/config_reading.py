import json
from typing import List


def read_and_validate_processing_json():
  """
  Reads the processing configurations JSON file, validates the keys and their types, and returns the retrieved values.

  Args:

  Returns:
    dict: A dictionary with the retrieved values from the JSON.

  Raises:
    ValueError: If a key is missing in the JSON.
    TypeError: If a value's type is incorrect.
  """
  # Define the expected types
  expected_types = {
    "save_path": str,
    "validation_proportion": float,
  }
  
  # Read the JSON from the file
  with open("processing_config.json", 'r') as file:
    data = json.load(file)
  
  # Validate each key
  for key, expected_type in expected_types.items():
    if key not in data:
      raise ValueError(f"Missing key: {key}")
    if not isinstance(data[key], expected_type):
      raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
  
  # Return the retrieved values
  return {
    "save_path": data["save_path"],
    "validation_proportion": data["validation_proportion"],
  }

import json
from typing import List


def read_and_validate_training_json():
  """
  Reads the training configurations JSON file, validates the keys and their types, and returns the retrieved values.

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
    "seed": int,
    "save_path": str,
    "float_precision": str,
    "batch_sizes": List[int],  # Updated to expect a list of integers
    "num_neurons": int,
    "learning_rate": float,
    "max_epochs": int
  }

  # Read the JSON from the file
  with open("training_config.json", 'r') as file:
    data = json.load(file)

  # Validate each key
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

  # Return the retrieved values
  return {
    "model_name": data["model_name"],
    "seed": data["seed"],
    "save_path": data["save_path"],
    "float_precision": data["float_precision"],
    "batch_sizes": data["batch_sizes"],
    "num_neurons": data["num_neurons"],
    "learning_rate": data["learning_rate"],
    "max_epochs": data["max_epochs"]
  }