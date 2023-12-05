from torch import nn


class MLP(nn.Module):
    r"""Flexible Multi-layer perceptron model, with optional batchnorm layers.

    Args:
        hidden_layers (list): List of layer dimensions, from input layer to output
            layer. If first input size is -1, will use a lazy layer.
        bias (boolean, optional): If set to :obj:`True`, bias will be used in linear
            layers. (default: :obj:`True`).
        activation (torch.nn.Module, optional): Activation function. (default:
            :obj:`nn.ReLU`).
        batchnorm (boolean, optional): If set to :obj:`True`, batchnorm layers are
            added after each linear layer, before the activation (default:
            :obj:`False`).
        drop_last_nonlin (boolean, optional): If set to :obj:`True`, the last layer
            won't have activations or batchnorm layers. (default: :obj:`True`)

    Examples:
        >>> m = MLP([-1, 16, 64])
        MLP(
          (layers): Sequential(
            (0): LazyLinear(in_features=0, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
          )
        )
    """

    def __init__(
        self,
        hidden_layers,
        *,
        bias=True,
        activation=nn.ReLU(True),
        batchnorm=False,
        drop_last_nonlin=True
    ):
        super().__init__()

        # build the layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            if in_dim == -1:
                layers.append(nn.LazyLinear(out_dim, bias=bias and not batchnorm))
            else:
                layers.append(nn.Linear(in_dim, out_dim, bias=bias and not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=out_dim, momentum=0.99))
                # ayers.append(nn.LayerNorm(out_dim))
            if activation is not None:
                activation = nn.PReLU(1)
                layers.append(activation)

        # remove activation and/or batchnorm layers from the last block
        if drop_last_nonlin:
            remove_layers = -(int(activation is not None) + int(batchnorm))
            if remove_layers:
                layers = layers[:remove_layers]

        self.layers = nn.Sequential(*layers)
        self.out_dim = hidden_layers[-1]

    def forward(self, x):
        x = self.layers(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
