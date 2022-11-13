import random
import numpy as np
import torch
import logging

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)    # set seed for numpy
    torch.manual_seed(seed) # set seed for cpu
    torch.cuda.manual_seed(seed) # set seed for current gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True # set deterministic for cudnn


"""
Logger
"""
def get_logger(name: str, file_path: str, stream=False, level='info')-> logging.RootLogger:

    level_map = {
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    
    logger = logging.getLogger(name)
    logger.setLevel(level_map[level])  # logging all levels
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger