"""Trainer

"""
from tqdm import tqdm
import torch
from torch.distributions import Normal
import torch.optim as optim
import numpy as np


class Trainer():
    """Trainer

    Attribues:
        model(object): 모델 객체
        optimizer (object): optimizer 객체
        scheduler (object): scheduler 객체
        loss_func (object): loss 함수 객체
        metric_funcs (dict): metric 함수 dict
        device (str):  'cuda' | 'cpu'
        logger (object): logger 객체
        loss (float): loss
        scores (dict): metric 별 score
    """

    def __init__(self,
                 opt,
                 rl_model,
                 classification_model,
                 train_loader,
                 env,
                 logger=None):

        self.opt = opt
        self.rl_model = rl_model
        self.classification_model = classification_model
        self.train_loader = train_loader
        self.env = env(opt=opt, classification_model=classification_model)
        self.logger = logger
        self.score = 0
        self.optimizer = optim.Adam(
            self.rl_model.parameters(), lr=opt.rl_learning_rate)

    def train(self):

        self.rl_model.train()  # Set model to training mode
        for original_images, _ in tqdm(self.train_loader):

            # Load data to gpu
            original_images = original_images.to(
                self.opt.device, dtype=torch.float)

            # Inference
            mu, std = self.rl_model(original_images)

            m_1 = Normal(mu[:, 0], std[:, 0])
            m_2 = Normal(mu[:, 1], std[:, 1])

            action_1 = m_1.sample()
            action_2 = m_2.sample()

            log_prob_1 = m_1.log_prob(action_1)
            log_prob_2 = m_2.log_prob(action_2)

            log_prob = log_prob_1 + log_prob_2

            action = torch.stack([action_1, action_2],
                                 dim=1)  # point, brightness
            r = self.env.step(original_images=original_images.type(
                torch.float32), action=action.cpu().numpy())

            for i in range(self.opt.batch_size):
                self.rl_model.put_data(
                    [r[i], log_prob[i]]
                )  # transation = (reward, log_prob)

            self.score += np.sum(r)/self.opt.batch_size

            self.rl_model.train_net(self.opt)

    def clear_history(self):
        torch.cuda.empty_cache()
        self.scores = 0
