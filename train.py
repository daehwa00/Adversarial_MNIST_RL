from time import time
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
    train_serial = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    seed_all(opt.seed)

    # Set device(GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device

    # Create train result directory and set Loggers
    train_result_dir = os.path.join(prj_dir, 'results', 'train', train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Set train logger
    logger = get_logger(name='train',
                        file_path=os.path.join(train_result_dir, 'train.log'),
                        level='info')

    # load train_loader and test_loader
    train_loader, test_loader = MNIST_data.create_dataloader(opt)

    # load reinforce model
    adversarial_model = Reinforce(opt).to(opt.device)

    # load cnn(classification) model
    classification_model = CNN(opt).to(opt.device)

    # load classification model parameter
    if opt.use_existing_classification_model:
        classification_model.load_state_dict(torch.load(
            "./models/pretrained_model/classification_model.pt"))
    else:
        # train classification model
        pass

        # Set Trainer
    trainer = Trainer(opt=opt, rl_model=adversarial_model, classification_model=classification_model, train_loader=train_loader,
                      env=MnistEnv)

    # Set recorder
    recorder = Recorder(record_dir=train_result_dir,
                        model=adversarial_model,
                        logger=logger)
    logger.info("Load early stopper, recorder")

    for epoch_id in range(opt.num_epochs):

        # Initiate result row
        row = dict()
        row['epoch_id'] = epoch_id
        row['train_serial'] = train_serial

        print(f"Epoch {epoch_id}/{opt.num_epochs} Train..")
        logger.info(f"Epoch {epoch_id}/{opt.num_epochs} Train..")
        tic = time()
        trainer.train()
        toc = time()
        row['train_score'] = trainer.score

        row['train_elapsed_time'] = round(toc-tic, 1)
        # Clear
        trainer.clear_history()

        # Performance record - row
        recorder.add_row(row)

        # Performance record - plot
        recorder.save_plot()
