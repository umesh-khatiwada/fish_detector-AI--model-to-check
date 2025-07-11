import json

def load_class_mapping(mapping_path):
    with open(mapping_path) as data:
        mappings = json.load(data)
    return {item['model_idx']: item['class_name'] for item in mappings}
