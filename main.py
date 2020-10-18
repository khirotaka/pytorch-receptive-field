import torch
import torch.nn as nn
from torch_receptive_field import receptive_field, receptive_field_for_unit


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 6, 3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(6, 6, 3, dilation=2),
            nn.ReLU(),
            nn.Conv1d(6, 6, 3, dilation=3),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(y)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # PyTorch v0.4.0
model = Net().to(device)

receptive_field_dict = receptive_field(model, (3, 100))
# receptive_field_for_unit(receptive_field_dict, "2", (1, 1))
print(receptive_field_dict["1"]["r"])

