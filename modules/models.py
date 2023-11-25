from torch import nn


class MLP(nn.Module):
    def __init__(
            self, 
            in_size: int, 
            out_size: int, 
            hidden_size: int, 
            hidden_width: int, 
            activation=nn.ReLU
        ):
        
        super(MLP, self).__init__()

        assert(in_size > 0)
        assert(out_size > 0)
        assert(hidden_size > 0)
        assert(hidden_width > 0)

        layers = []
        layers.append(nn.Linear(in_size, hidden_width))
        layers.append(activation())

        for _ in range(hidden_size):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(activation())

        layers.append(nn.Linear(hidden_width, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)