import os.path

import torch
from torch import nn
from nits.model import *
from nits.fc_model import *
import numpy as np
from utils import get_preprocessed_data, create_batcher, get_args
from train_utils import AverageMeter
import sklearn
from scipy import stats

class Trainer(nn.Module):
    def __init__(self, dataset_name, config={}, seed=1, additional_pseudo_samples=None,
                 global_data_premutations=None, model_num=0):
        super(Trainer, self).__init__()
        self.args = get_args()
        ######HP#####
        for key, val in config.items():
            self.args.__setattr__(key, val)

        self.args.dataset_name = dataset_name

        self.device = self.args.device if torch.cuda.is_available() else "cpu"
        self.seed = seed
        self.model_num = model_num
        data = get_preprocessed_data(self.args, dataset_name, seed)
        num_features = data.trn.x.shape[1]

        if global_data_premutations is None:
            self.data_permutations = Trainer.get_data_perumutations(self.args.num_ensemble_models, num_features,
                                                                    self.args.permute_model_input)
        else:
            self.data_permutations = global_data_premutations



        if additional_pseudo_samples is not None:
            data.trn.x = np.concatenate([data.trn.x, additional_pseudo_samples], axis=0)
            data.trn.y = np.concatenate([data.trn.y, np.zeros([len(additional_pseudo_samples), 1])], axis=0)

        self.data = data
        d = data.trn.x.shape[1]

        self.unique_feature_vals = {}
        for feature_num in range(d):
            self.unique_feature_vals[feature_num] = np.unique(self.data.trn.x[:, feature_num], axis=0)

        if self.args.variable_batch_size_factor > 1:
            self.batch_size = np.max([np.min([np.ceil(data.trn.x.shape[0] / self.args.variable_batch_size_factor),
                                              self.args.batch_size]), self.args.min_batch_size]).astype(np.int)
        else:
            self.batch_size = self.args.batch_size

        self.create_optimizer_schedualer(d, data)

        self.cdf_loss = nn.MSELoss(reduction="mean")
        self.ll_recon_loss = nn.MSELoss(reduction="mean")
        self.recon_loss = nn.MSELoss(reduction="mean")

        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)

    def create_optimizer_schedualer(self, d, data):
        self.model_list = []
        self.optim_list = []
        self.scheduler_list = []
        for _ in range(self.args.num_ensemble_models):
            self.model_list.append(self.create_model(d))

            if self.args.optimizer == 'adam':
                self.optim_list.append(torch.optim.Adam(self.model_list[-1].parameters(), lr=self.args.learning_rate))
            elif self.args.optimizer == 'sgd':
                self.optim_list.append(
                    torch.optim.SGD(self.model_list[-1].parameters(), lr=self.args.learning_rate, momentum=0.9,
                                    weight_decay=5e-4))
            else:
                raise f'optimizer "{self.args.optimizer}" not implemented'

            if self.args.scheduler == 'one_cycle':
                self.scheduler_list.append(
                    torch.optim.lr_scheduler.OneCycleLR(self.optim_list[-1], max_lr=self.args.learning_rate,
                                                        steps_per_epoch=int(np.ceil(len(data.trn.x) / self.batch_size)),
                                                        epochs=self.args.epochs))
            elif self.args.scheduler == 'cos_annealing':
                self.scheduler_list.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_list[-1],
                                                                                      T_max=int(np.ceil(len(data.trn.x) / self.batch_size)) * self.args.epochs))#
            else:
                raise NotImplementedError
            # self.scheduler_list.append(torch.optim.lr_scheduler.StepLR(self.optim_list[-1], step_size=1000, gamma=self.args.gamma))

    @staticmethod
    def get_data_perumutations(num_ensemble_models, num_features, permute_model_input):
        data_permutations = [np.array(range(num_features))]

        if num_ensemble_models > 1:
            for _ in range(num_ensemble_models - 1):
                if permute_model_input:
                    data_permutations.append(np.random.permutation(num_features))
                else:
                    data_permutations.append(np.array(range(num_features)))
        return data_permutations

    def create_model(self, d):
        max_val = 5
        min_val = -5
        self.max_val, self.min_val = torch.tensor(max_val).to(self.device).float(), torch.tensor(min_val).to(
            self.device).float()
        self.max_val *= self.args.bound_multiplier
        self.min_val *= self.args.bound_multiplier
        ######
        use_batch_norm = False
        zero_initialization = True
        weight_norm = False
        ######
        if len(self.args.num_reduce_dims):
            nits_input_dim = self.args.num_reduce_dims
        else:
            nits_input_dim = [1]

        nits_model = NITS(d=int(nits_input_dim[-1] * d), start=min_val, end=max_val, monotonic_const=1e-5,
                          A_constraint='neg_exp', arch=[1] + self.args.nits_arch,
                          final_layer_constraint='softmax',
                          add_residual_connections=self.args.add_residual_connections,
                          normalize_inverse=(not self.args.dont_normalize_inverse),
                          softmax_temperature=False).to(self.device)
        model = ResMADEModel(
            d=d,
            rotate=self.args.rotate,
            nits_model=nits_model,
            n_residual_blocks=self.args.n_residual_blocks,
            hidden_dim=self.args.hidden_dim,
            dropout_probability=self.args.dropout,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            weight_norm=weight_norm,
            scarf=self.args.scarf,
            unique_feature_vals=self.unique_feature_vals,
            nits_input_dim=nits_input_dim
        ).to(self.device)
        shadow = ResMADEModel(
            d=d,
            rotate=self.args.rotate,
            nits_model=nits_model,
            n_residual_blocks=self.args.n_residual_blocks,
            hidden_dim=self.args.hidden_dim,
            dropout_probability=self.args.dropout,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            weight_norm=weight_norm,
            scarf=self.args.scarf,
            unique_feature_vals=self.unique_feature_vals,
            nits_input_dim=nits_input_dim
        ).to(self.device)

        shadow = shadow.to(self.device)
        model = model.to(self.device)

        return EMA(model, shadow, decay=self.args.polyak_decay).to(self.device)


    def train_and_test(self):
        ensemble_f1_list, scores_ll_list, ensemble_auc_score, lr_list = self.train()
        return ensemble_auc_score, ensemble_f1_list, self.model_list, lr_list, scores_ll_list

    def train(self):
        epoch = 1
        print_every = 50
        max_val_ll = -np.inf
        patience = self.args.patience

        train_ll_meters = [AverageMeter(f'train_ll_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        train_recon_meters = [AverageMeter(f'train_ll_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        train_cdf_meters = [AverageMeter(f'train_cdf_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        ema_val_ll_meters = [AverageMeter(f'ema_val_ll_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        ema_test_f1_meters = [AverageMeter(f'ema_test_f1_{ind}', ':6.3f') for ind in
                              range(self.args.num_ensemble_models)]
        # val_ll_meters = [AverageMeter(f'val_ll_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        test_f1_meters = [AverageMeter(f'test_f1_{ind}', ':6.3f') for ind in range(self.args.num_ensemble_models)]
        interim_ensemble_f1_list = []
        interim_ensemble_ll_list = []
        interim_ensemble_cdf_list = []
        lr_list = []
        max_val_ll = -np.inf
        patience = self.args.patience
        keep_training = [True] * self.args.num_ensemble_models

        while np.any(keep_training):
            for model in self.model_list:
                model.train()
            epoch_ll = []
            recon_train = []
            for ind, (model, optim, scheduler) in enumerate(
                    zip(self.model_list, self.optim_list, self.scheduler_list)):
                for i, x in enumerate(create_batcher(self.data.trn.x[:, self.data_permutations[ind]],
                                                     batch_size=self.batch_size, device=self.device)):
                    if keep_training[ind]:
                        if (epoch >= self.args.pre_train_epochs) or not self.args.scarf:
                            mask_train_phase = False
                        else:
                            mask_train_phase = True

                        ll, cdf_per_feature, mask, ll_aug, recon = model(x, mask_train_phase=mask_train_phase,
                                                                  num_reduce_dims=self.args.num_reduce_dims)

                        epoch_ll.append(ll)
                        recon_train.append(recon)
                        if self.args.scarf and mask_train_phase:
                            ll_recon_loss = self.ll_recon_loss(ll, ll_aug)
                        else:
                            ll_recon_loss = torch.tensor(0).to(self.device)

                        cdf_loss = 0
                        if len(self.args.num_reduce_dims):
                            recon_loss = self.recon_loss(recon, x)
                        else:
                            recon_loss = torch.tensor(0).to(self.device)

                        if epoch > self.args.pre_train_epochs:
                            ll_recon_loss *= 0

                        optim.zero_grad()
                        curr_ll_loss = -ll.mean() * self.args.ll_loss_lambda + \
                                       self.args.var_loss_lambda * ((ll - ll.mean(dim=0)) ** 2).mean() \
                                       + ll_recon_loss * self.args.ll_recon_loss_lambda + \
                                       recon_loss * self.args.recon_loss_lambda
                        (curr_ll_loss).backward()
                        optim.step()

                        scheduler.step()
                        if ind == 0:
                            lr_list.append(scheduler.get_last_lr())

                        model.update()
                        train_ll_meters[ind].update(curr_ll_loss.detach().cpu().numpy(), x[0].size(0))
                        # train_recon_meters[ind].update(recon_loss.detach().cpu().numpy(), x[0].size(0))
                        if self.args.cdf_loss_lambda > 0:
                            train_cdf_meters[ind].update(cdf_loss.detach().cpu().numpy(), x[0].size(0))
                        else:
                            train_cdf_meters[ind].update(cdf_loss, x[0].size(0))

            ensemble_ll_score = np.mean([meter.avg for meter in train_ll_meters])
            ensemble_train_cdf = np.mean([meter.avg for meter in train_cdf_meters])
            interim_ensemble_ll_list.append(ensemble_ll_score)
            interim_ensemble_cdf_list.append(ensemble_train_cdf)

            epoch += 1
            if epoch == self.args.pre_train_epochs:
                print('!!!!!!!!!! Pre Train Finished !!!!!!!!!!!!!!!!')

            if epoch % print_every == 0 or epoch == 1 or epoch >= self.args.epochs or patience == 0:
                print(f'epoch: {epoch}')
                recon_train = torch.cat(recon_train) if recon_train[0] is not None else None
                ensemble_auc_score, ensemble_f1_score, ema_val_ll_meters, scores_ll_list = \
                    self.test(torch.cat(epoch_ll), recon_train, ema_val_ll_meters)

                # early stopping
                if self.args.early_stopping:
                    ema_val_ll = np.mean([meter.avg for meter in ema_val_ll_meters])
                    if ema_val_ll > max_val_ll + 1e-7:
                        patience = self.args.patience
                        max_val_ll = ema_val_ll
                    else:
                        patience -= 1

                interim_ensemble_f1_list.append(ensemble_f1_score)

            if epoch >= self.args.epochs or patience == 0:
                keep_training = [False]

        return ensemble_f1_score, scores_ll_list, ensemble_auc_score, lr_list

    def test(self, ll_train, recon_train, ema_val_ll_meters=None):
        auc_score, f1_score = [], []
        pred_vec_list = []
        scores_ll_list = []
        with torch.no_grad():
            for ind, model in enumerate(self.model_list):
                model.eval()
                log_like = []
                recon = []
                for i, samples in enumerate(create_batcher(self.data.tst.x[:, self.data_permutations[ind]],
                                                           batch_size=self.batch_size, device=self.device, phase='test')):
                    curr_log_like, _, curr_recon = model.shadow.forward_vec(samples, num_reduce_dims=self.args.num_reduce_dims)
                    log_like.append(curr_log_like)
                    if recon_train is not None:
                        recon.append((curr_recon - samples)**2)
                log_like = torch.cat(log_like)
                num_of_anomalies = int(self.data.tst.y.sum())

                curr_score_ll = (log_like.detach().cpu().numpy()).mean(axis=1)

                if ema_val_ll_meters is not None:
                    ema_val_ll_meters[ind].update(curr_score_ll, samples.size(0))

                if ll_train is not None:
                    score_all = torch.cat([ll_train, log_like]).detach().cpu().numpy()
                else:
                    score_all = log_like.detach().cpu().numpy()

                if recon_train is not None:
                    recon_all = torch.cat([recon_train.mean(dim=1, keepdim=True),
                                           torch.cat(recon).mean(dim=1, keepdim=True)]).detach().cpu().numpy()
                    torch.save(recon_all,
                               os.path.join(self.args.save_path,
                                            f'recon_{self.args.dataset_name}_seed_{self.seed}_model_num_{self.model_num}'))

                torch.save(score_all,
                           os.path.join(self.args.save_path,
                                        f'raw_ll_{self.args.dataset_name}_seed_{self.seed}_model_num_{self.model_num}'))
                cols_to_keep = np.std(score_all, axis=0) >= 1e-3
                score_all = score_all[:, cols_to_keep]
                result_zs = stats.zscore(score_all, axis=0)
                u, s = np.linalg.eigh(np.cov(result_zs.T))
                ll_new_t = score_all @ np.abs(s[:, -1:])
                ll_new = np.mean(ll_new_t, axis=1)
                ll_new = ll_new[len(ll_train):]
                curr_score_ll = -ll_new
                ll_anomaly_ind = np.argsort(curr_score_ll)[-num_of_anomalies:]
                curr_pred_binary = np.zeros_like(curr_score_ll)
                curr_pred_binary[ll_anomaly_ind] = 1
                try:
                    curr_f1_score = sklearn.metrics.f1_score(self.data.tst.y, curr_pred_binary)
                    curr_auc_score = sklearn.metrics.roc_auc_score(self.data.tst.y, curr_score_ll)
                except:
                    curr_f1_score = -1
                    curr_auc_score = -1

                auc_score.append(curr_auc_score)
                f1_score.append(curr_f1_score)
                scores_ll_list.append(curr_score_ll)
                pred_vec_list.append(curr_pred_binary)

        if len(self.model_list) == 1:
            pred_vec = pred_vec_list[0]
            ensemble_auc_score = np.mean(auc_score)
        elif self.args.ensemble_method == 'isml':
            pred_vec, score_list = self.pred_isml(num_of_anomalies, scores_ll_list, self.data.tst.y)
            ensemble_auc_score = sklearn.metrics.roc_auc_score(self.data.tst.y, score_list)
        elif self.args.ensemble_method == 'majority':
            pred_vec = np.median(pred_vec_list, axis=0)
            ensemble_auc_score = np.mean(auc_score)
        else:
            raise NotImplementedError

        ensemble_f1_score = sklearn.metrics.f1_score(self.data.tst.y, pred_vec)

        print(f'ensemble F1: {ensemble_f1_score}, mean AUC: {ensemble_auc_score}')

        return ensemble_auc_score, ensemble_f1_score, ema_val_ll_meters, scores_ll_list


    @staticmethod
    def pred_isml(num_of_anomalies, pred_vec_list, y):
        pred_array = np.array(pred_vec_list).squeeze()
        _, eig_cov_mat = np.linalg.eig(np.cov(pred_array))
        ensemble_weights = np.abs(eig_cov_mat[:, 0])
        score_list = np.sum(np.expand_dims(ensemble_weights, axis=-1) * np.array(pred_array), axis=0)
        pred_ind = np.argsort(score_list)[-num_of_anomalies:]
        pred_vec = np.zeros_like(y)
        pred_vec[pred_ind] = 1
        return pred_vec, score_list

if __name__ == "__main__":
    trainer = Trainer("cardio")
    trainer.train_and_test()
