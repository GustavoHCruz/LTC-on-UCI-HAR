import json


def read_and_validate_json(file_path):
  """
  Reads a JSON file, validates the keys and their types, and returns the retrieved values.

  Args:
    file_path (str): The path to the JSON file.

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
    "validation_proportion": float,
    "batch_size": int | str,
    "num_neurons": int,
    "learning_rate": float,
    "max_epochs": int
  }
  
  # Read the JSON from the file
  with open(file_path, 'r') as file:
    data = json.load(file)
  
  # Validate each key
  for key, expected_type in expected_types.items():
    if key not in data:
      raise ValueError(f"Missing key: {key}")
    if not isinstance(data[key], expected_type):
      raise TypeError(f"Incorrect type for {key}: Expected {expected_type.__name__}, but got {type(data[key]).__name__}")
  
  # Return the retrieved values
  return {
    "model_name": data["model_name"],
    "seed": data["seed"],
    "save_path": data["save_path"],
    "float_precision": data["float_precision"],
    "validation_proportion": data["validation_proportion"],
    "batch_size": data["batch_size"],
    "num_neurons": data["num_neurons"],
    "learning_rate": data["learning_rate"],
    "max_epochs": data["max_epochs"]
  }
