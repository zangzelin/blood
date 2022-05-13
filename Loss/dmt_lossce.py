from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch import Tensor

class myloss_ce(nn.Module):
    def __init__(
        self
    ):
        super(myloss_ce, self).__init__()

        self.mse = MSELoss()
        self.ce = CrossEntropyLoss()

    def forward(self, latent_data:Tensor, input_data:Tensor):
        # print(kNN_data.sum())

        # loss = self.mse(input_data, latent_data)

        # print(latent_data.shape, input_data.shape)
        loss = self.ce(latent_data, input_data)

        # print(loss_push_away)
        return loss