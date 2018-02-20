import argparse
from time import time

import torch.optim as optim
from torch.autograd import Variable

from caser import Caser
from evaluation import evaluate_ranking
from interactions import Interactions
from losses import weighted_sigmoid_log_loss
from utils import *

import numpy as np
import torch
import os


class Recommender(object):
    def __init__(self,
                 n_iter=None,
                 batch_size=None,
                 l2=None,
                 neg_samples=None,
                 learning_rate=None,
                 teacher_model_path=None,
                 teacher_topk_path=None,
                 use_cuda=False,
                 st_model_args=None,
                 th_model_args=None,
                 lamda=None,
                 mu=None,
                 dynamic_samples=None,
                 dynamic_start_epoch=None,
                 K=None,
                 teach_alpha=None):

        # data related
        self.L = None
        self.T = None

        # model related
        self._num_items = None
        self._num_users = None
        self._teacher_net = None  # teacher model
        self._net = None          # student model

        # learning related
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._neg_samples = neg_samples

        self._use_cuda = use_cuda
        self._loss_func = None

        # rank evaluation related
        self.test_sequence = None
        self._candidate = dict()

        # ranking distillation related
        self._teach_alpha = teach_alpha
        self._lambda = lamda
        self._mu = mu
        self._dynamic_samples = dynamic_samples
        self._dynamic_start_epoch = dynamic_start_epoch
        self._K = K

        self._teacher_model_path = teacher_model_path
        self._teacher_topk_path = teacher_topk_path

        # model args
        self._student_model_args = st_model_args
        self._teacher_model_args = th_model_args

    @property
    def _teacher_initialized(self):
        return self._teacher_net is not None

    def _initialize_teacher(self, interactions):
        # initialize teacher model
        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self._teacher_net = gpu(Caser(self._num_users,
                                      self._num_items,
                                      self._teacher_model_args), self._use_cuda)
        # load teacher model
        if os.path.isfile(self._teacher_model_path):
            output_str = ("loading teacher model %s" % self._teacher_model_path)
            print(output_str)

            checkpoint = torch.load(self._teacher_model_path)
            self._teacher_net.load_state_dict(checkpoint['state_dict'])
            output_str = "loaded model %s (epoch %d)" % (self._teacher_model_path, checkpoint['epoch_num'])
            print(output_str)
        else:
            output_str = "no model found at %s" % self._teacher_model_path
            print output_str

        # set teacher model to evaluation mode
        self._teacher_net.eval()

    @property
    def _student_initialized(self):
        return self._net is not None

    def _initialize_student(self, interactions):

        self._num_items = interactions.num_items
        self._num_users = interactions.num_users

        self.test_sequence = interactions.test_sequences

        self._net = gpu(Caser(self._num_users,
                              self._num_items,
                              self._student_model_args), self._use_cuda)
        self._optimizer = optim.Adam(
            self._net.parameters(),
            weight_decay=self._l2,
            lr=self._learning_rate)

        self._loss_func = weighted_sigmoid_log_loss

    def fit(self, train, test, verbose=False):

        sequences = train.sequences.sequences
        targets = train.sequences.targets
        users = train.sequences.user_ids.reshape(-1, 1)

        self.L, self.T = train.sequences.L, train.sequences.T

        n_train = sequences.shape[0]

        output_str = 'total training instances: %d' % n_train
        print(output_str)

        if not self._teacher_initialized:
            self._initialize_teacher(train)
        if not self._student_initialized:
            self._initialize_student(train)

        # make teacher top-K ranking
        candidates = self._get_teacher_topk(sequences, users, targets, self._K, self._teacher_topk_path)

        # initialize static weight
        weight_static = np.array(range(1, self._K + 1), dtype=np.float32)
        weight_static = np.exp(-weight_static / self._lambda)
        weight_static = weight_static / np.sum(weight_static)

        weight_static = Variable(gpu(torch.from_numpy(weight_static), self._use_cuda)).unsqueeze(0)

        # initialize dynamic weight
        weight_warp = None

        # count number of parameters
        num_params = 0
        for param in self._net.parameters():
            num_params += param.view(-1).size()[0]
        print("Number of params: %d" % num_params)

        indices = np.arange(n_train)
        start_epoch = 0

        for epoch_num in range(start_epoch, self._n_iter):

            t1 = time()

            # set model to training model
            self._net.train()

            (users, sequences, targets), shuffle_indices = shuffle(users,
                                                                   sequences,
                                                                   targets,
                                                                   indices=True)

            indices = indices[shuffle_indices]  # keep indices for retrieval teacher's top-K ranking

            negative_samples = self._generate_negative_samples(users, train, n=self._neg_samples*self.T)

            dynamic_samples = self._generate_samples(users, n=self._dynamic_samples)

            sequences_tensor = gpu(torch.from_numpy(sequences),
                                   self._use_cuda)
            user_tensor = gpu(torch.from_numpy(users),
                              self._use_cuda)
            item_target_tensor = gpu(torch.from_numpy(targets),
                                     self._use_cuda)
            item_negative_tensor = gpu(torch.from_numpy(negative_samples),
                                       self._use_cuda)
            dynamic_sample_tensor = gpu(torch.from_numpy(dynamic_samples),
                                        self._use_cuda)

            epoch_loss = 0.0
            epoch_regular_loss = 0.0

            for minibatch_num, \
                (batch_indices,
                 batch_sequence,
                 batch_user,
                 batch_target,
                 batch_negative,
                 batch_dynamic) in enumerate(minibatch(indices,
                                                       sequences_tensor,
                                                       user_tensor,
                                                       item_target_tensor,
                                                       item_negative_tensor,
                                                       dynamic_sample_tensor,
                                                       batch_size=self._batch_size)):

                sequence_var = Variable(batch_sequence)
                user_var = Variable(batch_user)
                item_target_var = Variable(batch_target)

                item_negative_var = Variable(batch_negative)
                dynamic_sample_var = Variable(batch_dynamic)

                # retrieval teacher top-K ranking for given indices
                teacher_topk_var = Variable(gpu(torch.from_numpy(candidates[batch_indices, :]), self._use_cuda))

                # concatenate all variables to get predictions in one run
                items_var = torch.cat((item_target_var, item_negative_var, teacher_topk_var, dynamic_sample_var), 1)

                items_prediction = self._net(sequence_var,
                                             user_var,
                                             items_var)
                (positive_prediction,
                 negative_prediction,
                 teacher_topk_prediction,
                 dynamic_sample_prediction) = torch.split(items_prediction, [item_target_var.size(1),
                                                                             item_negative_var.size(1),
                                                                             teacher_topk_var.size(1),
                                                                             dynamic_sample_var.size(1)], dim=1)

                self._optimizer.zero_grad()

                # compute dynamic weight
                dynamic_weights = list()
                for col in range(self._K):
                    col_prediction = teacher_topk_prediction[:, col].unsqueeze(1)

                    _dynamic_weight = torch.sum(col_prediction < dynamic_sample_prediction, dim=1).float() / self._dynamic_samples
                    _dynamic_weight = torch.floor(self._num_items * _dynamic_weight)

                    dynamic_weight = F.tanh(self._mu * (_dynamic_weight - col))
                    dynamic_weight = torch.clamp(dynamic_weight, min=0.0)

                    dynamic_weights.append(dynamic_weight)
                weight_dynamic = torch.stack(dynamic_weights, 1)

                if epoch_num + 1 >= self._dynamic_start_epoch:
                    weight = weight_dynamic * weight_static
                    weight = F.normalize(weight, p=1, dim=1)
                else:
                    weight = weight_dynamic
                weight = weight.detach()

                loss, regular_loss = self._loss_func(positive_prediction,
                                                     negative_prediction,
                                                     teacher_topk_prediction,
                                                     weight,
                                                     self._teach_alpha)

                epoch_loss += loss.data[0]
                epoch_regular_loss += regular_loss.data[0]

                loss.backward()

                # assert False
                self._optimizer.step()

            epoch_loss /= minibatch_num + 1
            epoch_regular_loss /= minibatch_num + 1

            t2 = time()

            if verbose and (epoch_num + 1) % 10 == 0:
                precision, recall, ndcg, mean_aps = evaluate_ranking(self, test, train, k=[3, 5, 10])

                str_precs = "precisions=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in precision])
                str_recalls = "recalls=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in recall])
                str_ndcgs = "ndcgs=%.4f,%.4f,%.4f" % tuple([np.mean(a) for a in ndcg])

                output_str = "Epoch %d [%.1f s]\tloss=%.4f, regular_loss=%.4f, " \
                             "map=%.4f, %s, %s, %s[%.1f s]" % (epoch_num + 1, t2 - t1,
                                                               epoch_loss, epoch_regular_loss,
                                                               mean_aps, str_precs, str_recalls, str_ndcgs,
                                                               time() - t2)
                print(output_str)
            else:
                output_str = "Epoch %d [%.1f s]\tloss=%.4f, regular_loss=%.4f[%.1f s]" % (epoch_num + 1, t2 - t1,
                                                                                          epoch_loss, epoch_regular_loss,
                                                                                          time() - t2)
                print(output_str)

    def _generate_negative_samples(self, users, interactions, n):
        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[np.random.randint(len(x))]

        return negative_samples

    def _generate_samples(self, users, n):
        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[np.random.randint(len(x))]
        return negative_samples

    def _get_teacher_topk(self, sequences, users, targets, k, path):
        if os.path.isfile(path):
            return np.load(path)
        else:

            n_train = sequences.shape[0]
            indices = np.arange(n_train)

            # convert from numpy to PyTorch tensor
            all_item_ids = np.arange(self._num_items).reshape(1, -1)
            all_item_ids = torch.from_numpy(all_item_ids.astype(np.int64)).clone()
            all_item_var = Variable(gpu(all_item_ids, self._use_cuda))

            sequences_tensor = gpu(torch.from_numpy(sequences), self._use_cuda)
            user_tensor = gpu(torch.from_numpy(users), self._use_cuda)

            # teacher_topk results
            teacher_topk = np.zeros((n_train, k), dtype=np.int64)

            for (batch_indices, batch_sequence, batch_user, batch_target) in minibatch(indices,
                                                                                       sequences_tensor,
                                                                                       user_tensor,
                                                                                       targets,
                                                                                       batch_size=256):
                batch_size = batch_indices.shape[0]

                sequence_var = Variable(batch_sequence)
                user_var = Variable(batch_user)
                batch_all_item_var = all_item_var.repeat(batch_size, 1)

                teacher_prediction = self._teacher_net(sequence_var,
                                                       user_var,
                                                       batch_all_item_var).detach()

                _, tops = teacher_prediction.topk(k * 2, dim=1)  # return the topk by column
                tops = cpu(tops.data).numpy()

                new_tops = np.concatenate((batch_target, tops), axis=1)
                topks = np.zeros((batch_size, k), dtype=np.int64)

                for i, row in enumerate(new_tops):
                    _, idx = np.unique(row, return_index=True)  # remove ground-truth targets
                    topk = row[np.sort(idx)][self.T:k + self.T]
                    topks[i, :] = topk
                teacher_topk[batch_indices, :] = topks
            np.save(path, teacher_topk)
            return teacher_topk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--train_root', type=str, default='datasets/gowalla/test/train.txt')
    parser.add_argument('--test_root', type=str, default='datasets/gowalla/test/test.txt')
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--T', type=int, default=1)
    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--neg_samples', type=int, default=3)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    # distillation arguments
    parser.add_argument('--th_embedding_dim', type=int, default=100)
    parser.add_argument('--teacher_model_path', type=str, default='checkpoints/gcaser-100.pth.tar')
    parser.add_argument('--teacher_topk_path', type=str, default='checkpoints/gcaser_candidates-100-top10.npy')
    parser.add_argument('--teach_alpha', type=float, default=1.0)
    parser.add_argument('--K', type=int, default=10)

    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--mu', type=float, default=0.1)
    parser.add_argument('--dynamic_samples', type=int, default=100)
    parser.add_argument('--dynamic_start_epoch', type=int, default=50)

    config = parser.parse_args()

    # model dependent arguments
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--d', type=int, default=50)

    # Caser args
    model_parser.add_argument('--nv', type=int, default=2)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='iden')
    model_parser.add_argument('--ac_fc', type=str, default='sigm')

    th_model_config = model_parser.parse_args()
    th_model_config.L = config.L
    th_model_config.d = config.th_embedding_dim

    st_model_config = model_parser.parse_args()
    st_model_config.L = config.L

    # set seed
    set_seed(config.seed,
             cuda=config.use_cuda)

    train = Interactions(config.train_root)
    # call to_sequence first, it will change item_map
    train.to_sequence(config.L, config.T)

    test = Interactions(config.test_root,
                        user_map=train.user_map,
                        item_map=train.item_map)

    print(config)
    print(st_model_config)

    # fit model
    model = Recommender(n_iter=config.n_iter,
                        batch_size=config.batch_size,
                        l2=config.l2,
                        neg_samples=config.neg_samples,
                        learning_rate=config.learning_rate,
                        teacher_model_path=config.teacher_model_path,
                        teacher_topk_path=config.teacher_topk_path,
                        use_cuda=config.use_cuda,
                        th_model_args=th_model_config,
                        st_model_args=st_model_config,
                        lamda=config.lamda,
                        mu=config.mu,
                        dynamic_samples=config.dynamic_samples,
                        dynamic_start_epoch=config.dynamic_start_epoch,
                        K=config.K,
                        teach_alpha=config.teach_alpha)

    model.fit(train, test, verbose=True)
