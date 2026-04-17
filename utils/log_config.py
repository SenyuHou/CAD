import os
import logging

def setup_logger(args, log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'logs')
    
    os.makedirs(log_dir, exist_ok=True)

    log_file_name = args.log_name + '.log'
    log_file_path = os.path.join(log_dir, log_file_name)

    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
