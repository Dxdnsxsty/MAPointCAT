import os
import torch
import torch.nn as nn

from .pu_net import PUNet
from ..drop_points import SORDefense


class DUPNet(nn.Module):
    def __init__(self, sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4):
        super(DUPNet, self).__init__()

        self.npoint = npoint
        self.sor = SORDefense(k=sor_k, alpha=sor_alpha)

        self.pu_net = PUNet(
            npoint=self.npoint,
            up_ratio=up_ratio,
            use_normal=False,
            use_bn=False,
            use_res=False,
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(current_dir, "pu-in_1024-up_4.pth")

        self.pu_net.load_state_dict(
            torch.load(weight_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pu_net = self.pu_net.cuda()
        self.pu_net.eval()

    def forward(self, x):
        with torch.enable_grad():
            x = self.sor(x)
            x = x.transpose(1, 2)
            x = self.pu_net(x)
            x = x.transpose(1, 2)
        return x
