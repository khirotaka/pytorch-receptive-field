# pytorch-receptive-field
 
[![Build Status](https://travis-ci.com/Fangyh09/pytorch-receptive-field.svg?branch=master)](https://travis-ci.com/Fangyh09/pytorch-receptive-field)

Compute CNN receptive field size in pytorch


## Usage
`git clone https://github.com/Fangyh09/pytorch-receptive-field.git`

```python
from torch_receptive_field import receptive_field
receptive_field(model, input_size=(channels, H, W))
```

Or
```python
from torch_receptive_field import receptive_field
dict = receptive_field(model, input_size=(channels, Seq_Len))
receptive_field_for_unit(receptive_field_dict, "2", (2,2))
```

## Example
```python
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
receptive_field_for_unit(receptive_field_dict, "2", (1, 1))

```
```
------------------------------------------------------------------------------
        Layer (type)    map size      start       jump receptive_field 
==============================================================================
        0               [3, 100]        0.5        1.0             1.0 
        1                [6, 98]        1.5        1.0             3.0 
        2                [6, 98]        1.5        1.0             3.0 
        3                [6, 94]        2.5        1.0             7.0 
        4                [6, 94]        2.5        1.0             7.0 
        5                [6, 88]        3.5        1.0            13.0 
        6                [6, 88]        3.5        1.0            13.0 
==============================================================================
Receptive field size for layer 2, unit_position (1, 1),  is 
 [(1.0, 3), (1.0, 4.0)]
```

## More
`start` is the center of first item in the map grid .

`jump` is the distance of the adjacent item in the map grid.

`receptive_field` is the field size of the item in the map grid.


## Todo
- [x] Add Travis CI 
  

## Related
Thanks @pytorch-summary

https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

