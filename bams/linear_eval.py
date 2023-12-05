import numpy as np
import torch
from sklearn import metrics

from sklearn.metrics import confusion_matrix


def train_linear_classfier(
    num_classes, train_data, test_data, device, lr=1e-2, weight_decay=1e-5
):
    r"""
    Trains a linear layer on top of the representations.
    """

    def train(classifier, train_data, optimizer):
        classifier.train()
        x, label = train_data

        # writer = SummaryWriter(comment='debug')
        x, label = x.to(device), label.to(device).squeeze()
        _, class_counts = torch.unique(label, return_counts=True)
        class_weights = 1 / class_counts

        sample_weights = class_weights[label]

        for step in range(1000):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)
            # loss and backprop
            loss = criterion(pred_logits, label)
            loss = sample_weights * loss
            loss = loss.sum() / sample_weights.sum()
            loss.backward()
            optimizer.step()
            # writer.add_scalar('loss', loss.item(), global_step=step)
        # writer.close()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            _, pred_class = torch.max(pred_logits, dim=1)
            pred_class = pred_class.cpu().numpy()

        f1_score = metrics.f1_score(label, pred_class, average="weighted")
        cm = confusion_matrix(label, pred_class, normalize="true")
        return f1_score, cm

    num_feats = train_data[0].size(1)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(
        0, unbiased=False, keepdim=True
    )
    train_data[0] = train_data[0] - mean
    test_data[0] = test_data[0] - mean

    train_data[0] = train_data[0] / torch.norm(train_data[0], dim=1, keepdim=True)
    test_data[0] = test_data[0] / torch.norm(test_data[0], dim=1, keepdim=True)

    classifier = torch.nn.Linear(num_feats, num_classes).to(device)
    optimizer = torch.optim.AdamW(
        params=classifier.parameters(), lr=lr, weight_decay=weight_decay
    )

    train(classifier, train_data, optimizer)
    test_f1, test_cm = test(classifier, test_data)

    return test_f1, test_cm


def train_linear_regressor(train_data, test_data, device, lr=1e-4, weight_decay=1e-4):
    r"""
    Trains a linear layer on top of the representations.
    """

    def train(regressor, train_data, optimizer):
        regressor.train()
        x, label = train_data

        x, label = x.to(device), label.to(device).squeeze()

        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred = regressor(x).squeeze()
            # loss and backprop
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

    def test(regressor, data):
        regressor.eval()
        x, label = data
        x, label = x.to(device), label.to(device).squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred = regressor(x.to(device)).squeeze()
            mse = torch.nn.functional.mse_loss(pred, label).cpu().numpy()

        return mse

    num_feats = train_data[0].size(1)
    criterion = torch.nn.MSELoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(
        0, unbiased=False, keepdim=True
    )
    train_data[0] = (train_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    regressor = torch.nn.Linear(num_feats, 1).to(device)
    optimizer = torch.optim.AdamW(
        params=regressor.parameters(), lr=lr, weight_decay=weight_decay
    )

    train(regressor, train_data, optimizer)
    test_mse = test(regressor, test_data)

    return test_mse


def compute_representations(encoder, x, device):
    x = x.to(device)
    with torch.no_grad():
        repr = encoder(x)
    return repr
