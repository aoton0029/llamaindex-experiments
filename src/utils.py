import yaml

def yaml_to_dict(yaml_str:str) -> dict:
    try:
        data = yaml.safe_load(yaml_str)
        return data if isinstance(data, dict) else (data or {})
    except yaml.YAMLError as e:
        raise ValueError(e)
