from collections import OrderedDict

import torch
import torch.nn as nn

from bams.models import TemporalConvNet, MLP


class BAMS(nn.Module):
    r"""BAMS model.

    Args:
        input_size (int): Number of input features.
        predictor (dict): Parameters for the predictor MLP.
        encoders (dict[dict]): A dictionnary of encoders, where each key is the name of
            the encoder, and each value is a dictionnary of parameters for the encoder.
            Each encoder is a TemporalConvNet.
    """

    def __init__(
        self,
        input_size,
        *,
        predictor=None,
        **encoder_kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.representation_size = 0

        encoders = dict()
        for name, tcn_kwargs in encoder_kwargs.items():
            assert "num_inputs" not in tcn_kwargs
            encoders[name] = TemporalConvNet(num_inputs=input_size, **tcn_kwargs)
            self.representation_size += tcn_kwargs["num_channels"][-1]

        self.encoders = torch.nn.ModuleDict(encoders)

        # hoa predictor (first layer is a lazy linear layer)
        self.predictor = MLP(**predictor)

        # byol predictors
        byol_predictors = dict()
        for name, tcn_kwargs in encoder_kwargs.items():
            emb_dim = tcn_kwargs["num_channels"][-1]
            byol_predictors[name] = nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 4, bias=False),
                nn.BatchNorm1d(emb_dim * 4, eps=1e-5, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim * 4, emb_dim, bias=True),
            )
        self.byol_predictors = torch.nn.ModuleDict(byol_predictors)

    def forward(self, x):
        # input shape: (B: batch_size, L:sequence_length, N: num_feats)
        # forward through TCNs
        embs = OrderedDict()
        byol_preds = OrderedDict()
        for name, encoder in self.encoders.items():
            embs[name] = encoder(x)  # (B, L, N)
            flattened_emb = embs[name].flatten(0, 1)  # (B*L, N)
            pred_emb = self.byol_predictors[name](flattened_emb)
            byol_preds[name] = pred_emb.reshape(embs[name].shape)

        # concatenate embeddings
        h = torch.cat(list(embs.values()), dim=2)  # (B, L, N)

        # concatenate input and embeddings
        hx = torch.cat([h, x], dim=2)
        # prediction
        hoa_pred = self.predictor(hx)
        return embs, hoa_pred, byol_preds

    def __repr__(self) -> str:
        args = [
            f"  {name}: {encoder.__class__.__name__}"
            f" (receptive field: {encoder.receptive_field},"
            f" feature dim: {encoder.feat_dim})"
            for name, encoder in self.encoders.items()
        ]
        args.append(
            f"  predictor: {self.predictor.__class__.__name__}"
            f" (input size: {self.input_size},"
            f" output size: {self.predictor.out_dim})"
        )
        return "{}([\n{}\n])".format(self.__class__.__name__, ",\n".join(args))
