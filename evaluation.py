import torch
from torch.autograd import Variable
import numpy as np
from utils import cpu, gpu, minibatch


def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def _compute_ndcg(targets, predictions, k):
    k = min(len(targets), k)

    if len(predictions) > k:
        predictions = predictions[:k]

    # compute idcg
    idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
    dcg = 0.0
    for i, p in enumerate(predictions):
        if p in targets:
            dcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg

    return ndcg


def _get_predictions(model):

    model._net.eval()

    sequences = model.test_sequence.sequences
    users = np.arange(model._num_users)

    n_train = sequences.shape[0]
    indices = np.arange(n_train)

    # convert from numpy to PyTorch tensor
    all_item_ids = np.arange(model._num_items).reshape(1, -1)
    all_item_ids = torch.from_numpy(all_item_ids.astype(np.int64)).clone()
    all_item_var = Variable(gpu(all_item_ids, model._use_cuda))

    sequences_tensor = gpu(torch.from_numpy(sequences), model._use_cuda)
    user_tensor = gpu(torch.from_numpy(users), model._use_cuda)

    top = 1000
    all_predictions = np.zeros((model._num_users, top), dtype=np.int64)

    for (batch_indices, batch_sequence, batch_user) in minibatch(indices,
                                                                 sequences_tensor,
                                                                 user_tensor,
                                                                 batch_size=256):
        batch_size = batch_user.shape[0]

        sequence_var = Variable(batch_sequence)
        user_var = Variable(batch_user)
        batch_all_item_var = all_item_var.repeat(batch_size, 1)

        prediction = model._net(sequence_var,
                                        user_var,
                                        batch_all_item_var)

        _, tops = prediction.topk(top, dim=1)
        all_predictions[batch_indices, :] = cpu(tops.data).numpy()

    return all_predictions


def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    all_predictions = _get_predictions(model)

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    ndcgs = [list() for _ in range(len(ks))]
    apks = list()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = all_predictions[user_id, :]

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

            ndcg = _compute_ndcg(targets, predictions, _k)
            ndcgs[i].append(ndcg)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]
        ndcgs = ndcgs[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, ndcgs, mean_aps


