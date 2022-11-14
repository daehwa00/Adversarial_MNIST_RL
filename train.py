import time
import os
import torch
from options.train_options import TrainOptions
import MNIST_data
from models.cnn_model import CNN
from models.reinforce_model import Reinforce
import datetime
import sys
from modules.trainer import Trainer
from modules.utils import seed_all, get_logger
from modules.recorders import Recorder
from env.mnist_env import MnistEnv


if __name__ == '__main__':
    opt = TrainOptions().parse()

    # get current directory
    prj_dir = os.path.dirname(os.path.abspath(''))
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

    # load reinforce model
    Reinforce_model = Reinforce(opt)

    # Set Trainer
    trainer = Trainer(opt, Reinforce_model, train_loader, test_loader,
                      MnistEnv, logger, train_result_dir)

    # Set recorder
    recorder = Recorder(record_dir=train_result_dir,
                        model=Reinforce_model,
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
        recorder.save_plot(opt.plot)
