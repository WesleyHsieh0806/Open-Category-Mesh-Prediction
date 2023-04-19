import logging 
import os 

def setup_logger(name, cfg, args):
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    os.makedirs(cfg.training.log_dir, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(cfg.training.log_dir, 'log{}.txt'.format(args.local_rank)), mode='w')
    fileHandler.setLevel(logging.INFO)

    # write output to log_dir/log.txt
    logger.addHandler(fileHandler)
    
    return logger