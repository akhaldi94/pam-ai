import yaml

def get_model_path(yaml_file_path):
    """
    Reads the YAML file and returns the model path.

    :param yaml_file_path: The path to the YAML file.
    :return: The model path as a string.
    :raises: FileNotFoundError if the file does not exist.
             KeyError if the expected keys are not found.
    """
    try:
        # Load the YAML file
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Access the Path value
        model_path = data['Model']['Path']
        return model_path

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {yaml_file_path} was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected key: {e}")
