import json

def to_dict(cls):
    # This function will extract both class and instance attributes.
    # It checks if the attribute belongs to the class (using `type(cls)` for class attributes).
    attrs = {attr: getattr(cls, attr) for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")}
    return attrs

def save_config_to_json(config_class, filename):
    # Convert the configuration class to a dictionary
    config_dict = to_dict(config_class)

    # Write the dictionary to a file as JSON
    with open(filename, 'w') as file:
        json.dump(config_dict, file, indent=4)