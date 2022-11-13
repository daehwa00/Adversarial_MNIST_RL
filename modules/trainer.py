"""Trainer

"""
from tqdm import tqdm
import torch
from torch.distributions import Normal

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
                model,
                optimizer,
                scheduler,
                loss_func,
                metric_funcs,
                device,
                env,
                logger=None):        
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.metric_funcs = metric_funcs
        self.device = device
        self.env = env
        self.logger = logger

        self.loss = 0
        self.scores = {metric_name: 0 for metric_name, _ in self.metric_funcs.items()}

    def train(self, dataloader, epoch_index=0):
        
        self.model.train()
        for original_image, _ in enumerate(tqdm(dataloader)):

            # Load data to gpu
            original_image = original_image.to(self.device, dtype=torch.float)
            
            # Inference
            mu, std = self.model(original_image)
            
            m_1 = Normal(mu[:, 0], std[:, 0])
            m_2 = Normal(mu[:, 1], std[:, 1])

            action_1 = m_1.sample()
            action_2 = m_2.sample()
            
            log_prob_1 = m_1.log_prob(action_1)
            log_prob_2 = m_2.log_prob(action_2)
            
            log_prob = log_prob_1 + log_prob_2
            
            action = torch.stack([action_1, action_2], dim=1)  # point, brightness
            r = self.env.step(original_image.type(torch.float32), action.cpu().numpy())
            
            for i in range(opt.batch_size):
                self.model.put_data(
                    [r[i], log_prob[i]]
                )  # transation = (reward, log_prob)

            self.model.train_net()
        
    def clear_history(self):

        torch.cuda.empty_cache()
        self.loss = 0
        self.scores = {metric_name: 0 for metric_name, _ in self.metric_funcs.items()}
        self.logger.debug(f"TRAINER | Clear history, loss: {self.loss}, score: {self.scores}")