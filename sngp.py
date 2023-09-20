import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from rfgp import RandomFeatureGaussianProcess


class SNGP(nn.Module):
    def __init__(self,
                 input_features: int = 2,
                 num_hidden_layers: int = 6,
                 num_hidden: int = 128,
                 dropout_rate: float = 0.1,
                ) -> None:
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        
        self.input_layer = nn.Linear(in_features=input_features, out_features=num_hidden, bias=False)
        for p in self.input_layer.parameters():
            p.requires_grad = False

        self.hidden_layers = nn.ModuleList([spectral_norm(nn.Linear(num_hidden, num_hidden, bias=True)) for _ in range(num_hidden_layers)])

        self.gp_layer = RandomFeatureGaussianProcess(in_features=128, out_features=2, num_inducing=1024, gp_cov_momentum=-1)

    def forward(self, input, return_gp_cov=False, update_cov=False):
        output = self.input_layer(input)

        for hidden_layer in self.hidden_layers:
            residual = F.relu(hidden_layer(output))
            residual = F.dropout(residual, p=self.dropout_rate, training=self.training)
            output = output + residual

        return self.gp_layer(output, return_gp_cov=return_gp_cov, update_cov=update_cov)