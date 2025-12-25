import os
from tabnanny import verbose
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from box import ConfigBox
import base64
from pathlib import Path
from typing import Any

# Simple decorator to replace ensure_annotations
def ensure_annotations(func):
    """Simple replacement for ensure_annotations decorator"""
    return func

def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """ Reads a yaml file and returns
        Args:
            path_to_yaml (Path): Path to the yaml file
        Raises:
            ValueError: If the yaml file is empty
            e: Empty file
        Returns:
            ConfigBox: ConfigBox typr
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list):
    """create list of directories
    Args:
        path_to_directories (list): list of directories to be created
            ignore_log (bool, optional): ignore logging if True. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
      
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")  

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json data
    Args:
        path (Path): path to json file
    Raises:
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path, "r") as f:
            content = json.load(f)
            logger.info(f"json file loaded successfully from: {path}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("json file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary data
    Args:
        data (Any): data to be saved
        path (Path): path to the file
    """

    joblib.dump(data, path)
    logger.info(f"binary file saved at: {path}")    

@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data
    Args:
        path (Path): path to the file

    Returns:
        Any: data loaded from the file
    """

    data = joblib.load(path)
    logger.info(f"binary file loaded successfully from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    Args:
        path (Path): path to the file
    Returns:
        str: size in KB
    """

    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def decodeImage(imgstring, fileName):
    """decode base64 string to bytes
    Args:
        image_base64 (str): base64 string
    Returns:
        bytes: decoded bytes
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

@ensure_annotations
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

