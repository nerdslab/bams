import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

def compute_representations(model, dataset, device, batch_size=4):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embs_dict = defaultdict(list)
    for data in loader:
        x = data['input'].float().to(device)

        with torch.no_grad():
            embs, _, _ = model(x)
            for key, emb in embs.items():
                embs_dict[key].append(emb.detach().cpu())

    embs = {key: torch.cat(emb_list) for key, emb_list in embs_dict.items()}
    return embs