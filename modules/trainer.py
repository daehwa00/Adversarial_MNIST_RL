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
                 adversarial_model,
                 classification_model,
                 train_loader,
                 optimizer,
                 env,
                 logger=None):

        self.opt = opt
        self.adversarial_model = adversarial_model
        self.classification_model = classification_model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.env = env(opt=opt, classification_model=classification_model)
        self.logger = logger
        self.score = 0
        self.data = []

    def train(self, adversarial_model):
        for original_images, _ in tqdm(self.train_loader):
            # Load data to gpu
            original_images = original_images.to(
                self.opt.device, dtype=torch.float)

            # Inference
            mu, std = adversarial_model(original_images)

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
                self.put_data([r[i], log_prob[i]])

            self.optimizer.zero_grad()
            r_lst, log_prob_lst = [], []
            for transition in self.data:
                r_lst.append(transition[0])
                log_prob_lst.append(transition[1])
            r_lst = torch.tensor(r_lst).to(self.opt.device)
            log_prob_lst = torch.stack(log_prob_lst).to(self.opt.device)
            log_prob_lst = (-1) * log_prob_lst
            loss = log_prob_lst * r_lst
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.data = []

            # score는 reward의 평균
            self.score += np.mean(r)
        self.score = self.score / 20

    def clear_history(self):
        torch.cuda.empty_cache()
        self.score = 0

    def put_data(self, transition):
        self.data.append(transition)
