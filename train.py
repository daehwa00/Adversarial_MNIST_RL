import time
from models import create_model
import os
import copy
import numpy as np
import torch, shutil
from options.train_options import TrainOptions
import MNIST_data
import models

import datetime

import sys

from modules.trainer import Trainer
from modules.optimizer import get_optimizer
from modules.utils import seed_all, get_logger

from env.mnist_env import MnistEnv

seed_all(3)

def make_val_opt(opt):

    val_opt = copy.deepcopy(opt)
    val_opt.preprocess = ''  #
    # hard-code some parameters for test
    val_opt.num_threads = 0   # test code only supports num_threads = 1
    val_opt.batch_size = 4    # test code only supports batch_size = 1
    val_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    val_opt.angle = 0
    val_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    val_opt.phase = 'val'
    val_opt.split = opt.val_split  # function in jsonDataset and ListDataset
    val_opt.isTrain = False
    val_opt.aspect_ratio = 1
    val_opt.results_dir = './results/'
    val_opt.dataroot = opt.val_dataroot
    val_opt.dataset_mode = opt.val_dataset_mode
    val_opt.dataset_type = opt.val_dataset_type
    val_opt.json_name = opt.val_json_name
    val_opt.eval = True

    val_opt.num_test = 2000
    return val_opt

if __name__ == '__main__':
    
    opt = TrainOptions().parse()

    # get current directory
    prj_dir = os.path.dirname(os.path.abspath(__file__))    
    sys.path.append(prj_dir)
    
    # Set train serial (cureent time)
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    seed_all(opt.seed)
    
    # Set device(GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    
    # Create train result directory and set Loggers
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Set train logger
    logging_level = 'debug' if opt['verbose'] else 'info'
    logger = get_logger(name='train',
                    file_path=os.path.join(train_result_dir, 'train.log'),
                    level=logging_level)

    # load train_loader and test_loader
    train_loader, test_loader = MNIST_data.create_dataloader(opt)

    # load model
    model = models.REINFORCE(opt)
    
    # Set Trainer
    trainer = Trainer(opt, model, train_loader, test_loader, MnistEnv, logger, train_result_dir)
    
    # Set recorder
    recorder = Recorder(record_dir=train_result_dir,
                        model=model,
                        optimizer=opt.optimizer,
                        scheduler=opt.scheduler,
                        logger=logger)
    logger.info("Load early stopper, recorder")
    
    for epoch_id in range(opt.epoch_num):
        
        # Initiate result row
        row = dict()
        row['epoch_id'] = epoch_id
        row['train_serial'] = train_serial
        row['lr'] = trainer.scheduler.get_last_lr()

        print(f"Epoch {epoch_id}/{opt.epoch_num} Train..")
        logger.info(f"Epoch {epoch_id}/{opt.epoch_num} Train..")
        tic = time()
        trainer.train(dataloader=train_loader)
        toc = time()
        row['train_loss'] = trainer.loss
        for metric_name, metric_score in trainer.scores.items():
            row[f'train_{metric_name}'] = metric_score

        row['train_elapsed_time'] = round(toc-tic, 1)
        # Clear
        trainer.clear_history()
        
        # Performance record - row
        recorder.add_row(row)

        # Performance record - plot
        recorder.save_plot(config['plot'])