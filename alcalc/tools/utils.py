import yaml


def write_yaml(file_path, dictionary):
    """Writes a given dictionary to a YAML file."""
    with open(file_path, "w") as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)


def read_yaml(file_path):
    """Reads a YAML file and returns the contents as a dictionary."""
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)
