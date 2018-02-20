"""
Loss functions for recommender models.

The pointwise, BPR, and hinge losses are a good fit for
implicit feedback models trained through negative sampling.

The regression and Poisson losses are used for explicit feedback
models.
"""

import torch
import torch.nn.functional as F


def sigmoid_log_loss(positive_predictions, negative_predictions):

    loss1 = -torch.log(F.sigmoid(positive_predictions))
    loss0 = -torch.log(1 - F.sigmoid(negative_predictions))

    # loss = torch.cat((loss1.view(-1), loss0.view(-1))).mean()
    loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

    return loss.mean()


def weighted_sigmoid_log_loss(positive_predictions, negative_predictions, candidate_predictions, weight, alpha=1.0):

    loss1 = -torch.log(F.sigmoid(positive_predictions))
    loss0 = -torch.log(1 - F.sigmoid(negative_predictions))

    loss_cand = -torch.log(F.sigmoid(candidate_predictions))

    if weight is not None:
        loss_cand = loss_cand * weight.expand_as(loss_cand)

    if alpha is not None:
        loss_cand = loss_cand * alpha

    loss = torch.sum(torch.cat((loss1, loss0, loss_cand), 1), dim=1)
    reg_loss = torch.sum(torch.cat((loss1, loss0), 1), dim=1)

    return loss.mean(), reg_loss.mean()
