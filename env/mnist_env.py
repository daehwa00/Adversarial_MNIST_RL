import numpy as np
import torch

class MnistEnv():
    def __init__(self, classification_model, random=True):
        super().__init__()
        self.classification_model = classification_model  # pre-trained CNN model

    def step(self, original_image, action):  # action = [위치, 밝기]
        action = torch.tensor(action)
        action = torch.sigmoid(action)

        point = (action[:, 0] * 676).int()
        brightness = ((action[:, 1] * 255).int()) / 255

        arr = []
        center = [torch.div(point, 26, rounding_mode="trunc") + 1, point % 26 + 1]
        action = [point, brightness]

        for i in range(batch_size):
            changed_image = original_image[i].squeeze().squeeze().cpu().numpy()  #   [28,28]
            # Stamp 찍기
            changed_image[center[0][i], center[1][i]] = brightness[i]
            arr.append(changed_image)

        changed_images = torch.stack([torch.tensor(a).unsqueeze(0) for a in arr], dim=0)

        with torch.no_grad():
            original_outputs = self.classification_model(original_image.to(device))
            changed_outputs = self.classification_model(changed_images.to(device))

        rewards = torch.sum(
            torch.nn.functional.kl_div(
                original_outputs.log(), changed_outputs, size_average = None, reduction="none"
            ),
            dim=1,
        )

        # original_outputs와 changed_outputs의 max의 index가 같으면 reward를 -1/reward로 준다.
        rewards[original_outputs.max(1)[1] == changed_outputs.max(1)[1]] = (
            -1 / rewards[original_outputs.max(1)[1] == changed_outputs.max(1)[1]]
        )

        rewards[rewards < -1000] = -1000
        rewards[rewards > 1000] = 1000

        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1000, neginf=-1000) # nan을 0으로, inf를 100으로 바꾼다.

        return rewards.cpu().numpy()