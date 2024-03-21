from src.models import Model
import json

def load_config(file_path):
    """Load the contents of the config.json file to get the model files."""
    with open(file_path, 'r') as config_file:
        config_data = json.load(config_file)
        return config_data.get('model_path')

def load_model(config_file_path = "config/config.json"):

    model_path = load_config(config_file_path)
    model = Model(model_path)

    return model