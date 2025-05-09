import yaml


def update_config(file, config):
    """
    Updates a configuration object with values from a YAML file.
    :param file: ´str´ Path to the YAML file.
    :param config: ´Config´ Configuration object to be updated.
    """
    with open(file, "r", encoding="utf-8") as f:
        config_dict = yaml.load(f.read(), Loader=config.yaml_loader)
        if config_dict is not None:
            config.final_config_dict.update(config_dict)
    return config
