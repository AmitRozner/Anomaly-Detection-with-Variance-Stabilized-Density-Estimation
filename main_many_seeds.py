import time

import pandas as pd
import os
import numpy as np
import glob
from trainer import Trainer
import torch
import random
import multiprocessing
from multiprocessing import Pool, Process, JoinableQueue, Event
import argparse
from tqdm import tqdm
from utils import get_preprocessed_data, invert_permutation, list_str_to_list
import utils
import resource
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import sklearn

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
multiprocessing.set_start_method('spawn', force=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=str, default='amit')
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--hp_search', action="store_true")
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('--min_batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-p', '--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--pre_train_epochs', type=int, default=0)
    parser.add_argument('--cascade_training', action="store_true")
    parser.add_argument('--use_most_ano_examples', action="store_true")
    parser.add_argument('--single_dataset_name', type=str, default='')
    parser.add_argument('--num_ensemble_models', type=int, default=5)
    parser.add_argument('-rc', '--add_residual_connections', action="store_true")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--scheduler', type=str, choices=['one_cycle', "cos_annealing"], default='cos_annealing')
    parser.add_argument('--num_parallel_procs', type=int, default=2)
    parser.add_argument('--variable_batch_size_factor', type=int, default=1)
    parser.add_argument('--ensemble_method', type=str, default='isml', choices=['majority', 'isml'])
    parser.add_argument('--cdf_loss_lambda', type=float, default=0.0)
    parser.add_argument('--ll_loss_lambda', type=float, default=1.0)
    parser.add_argument('--var_loss_lambda', type=float, default=3.33333333)
    parser.add_argument('--recon_loss_lambda', type=float, default=0.0)
    parser.add_argument('--ll_recon_loss_lambda', type=float, default=0.0)
    parser.add_argument('-pd', '--polyak_decay', type=float, default=0.995)
    parser.add_argument('--early_stopping', action="store_true")
    parser.add_argument('-hi', '--hidden_dim', type=int, default=1024)
    parser.add_argument('-nr', '--n_residual_blocks', type=int, default=2)
    parser.add_argument('--num_training_rounds', type=int, default=2)
    parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,2]')
    parser.add_argument('--scarf', action="store_true")
    parser.add_argument('--num_reduce_dims', nargs='+', type=float, default=[])
    parser.add_argument('--num_reduce_dims_pca', type=int, default=0)
    parser.add_argument('--normalize_dataset', type=str, default='zscore')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--data_dir', type=str, default="./datasets")
    parser.add_argument('--permute_model_input', action="store_false")
    return parser.parse_args()


def main_func(dataset_name, seed=1, config={}, queue=None, global_data_premutations=None, device='cuda:0',
              additional_pseudo_samples=None, train_round=0, model_num=0):
    np.random.seed(int(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    config['device'] = device
    trainer = Trainer(dataset_name, config=config, seed=seed, global_data_premutations=global_data_premutations,
                      additional_pseudo_samples=additional_pseudo_samples, model_num=model_num)
    n_samples = trainer.data.trn.x.shape[0]
    ensemble_auc_score, ensemble_f1_score, model_list, lr_list, scores_ll_list = trainer.train_and_test()
    num_to_sample = int(np.max([np.min([n_samples / 50, 50]), 2]))
    return_models = []
    global_sampled_examples = []
    most_anomalous_trn_examples = []
    if train_round + 1 < config['num_training_rounds']:
        for curr_model_ind, model in enumerate(model_list):
            curr_sampled_examples = model.shadow.sample(num_to_sample, config['device']).detach().cpu()
            global_sampled_examples.append(curr_sampled_examples)
            train_samples = torch.tensor(trainer.data.trn.x[:, trainer.data_permutations[curr_model_ind]],
                                         device=trainer.device).float()
            ll_train, _, recon = model.shadow.forward_vec(train_samples)
            ll_train_ind = np.argsort((ll_train.detach().cpu().numpy()).sum(axis=1))
            most_anomalous_trn_examples.append(
                trainer.data.trn.x[:, trainer.data_permutations[curr_model_ind]][ll_train_ind[:num_to_sample]])
            return_models.append(model.shadow.cpu())

    if queue is None:
        return ensemble_auc_score, ensemble_f1_score, return_models, seed, global_sampled_examples, lr_list, \
               most_anomalous_trn_examples, scores_ll_list
    else:
        # return only if needed for cascade later
        queue.put((ensemble_auc_score, ensemble_f1_score, return_models, seed, global_sampled_examples, lr_list, \
                   most_anomalous_trn_examples, scores_ll_list))
        queue.join()


def run_one_ds_many_seeds(dataset_name, global_args):
    seeds = range(0, global_args.num_seeds)
    global_data_premutations, n_samples, y = data_premutations(dataset_name, global_args)
    global_sampled_examples = None

    for train_round in range(global_args.num_training_rounds):
        auc_list = []
        f1_list = []
        ensemble_f1_list = []
        ensemble_auc_list = []
        model_list = []
        proc_list = []
        scores_list = []
        queue_list = []
        most_anomalous_trn_examples_list = []

        if train_round < 2:
            global_sampled_examples_list = []

        local_args = vars(global_args).copy()
        local_args['num_ensemble_models'] = 1
        start = time.time()
        for ind, seed in tqdm(enumerate(seeds)):
            for model_num in range(global_args.num_ensemble_models):
                queue_list.append(JoinableQueue())
                proc_list.append(Process(target=main_func, args=(dataset_name, seed, local_args, queue_list[-1],
                                                                 [global_data_premutations[model_num]],
                                                                 global_args.device[
                                                                     model_num % len(global_args.device)],
                                                                 global_sampled_examples, train_round, model_num)))
                proc_list[-1].start()

                if (len(proc_list) >= global_args.num_parallel_procs):
                    proc_list, queue_list, scores_list = finish_proc(auc_list, dataset_name, f1_list, global_args,
                                                                     model_list, proc_list,
                                                                     queue_list, global_sampled_examples_list,
                                                                     f'round_{train_round}_single_model',
                                                                     most_anomalous_trn_examples_list, scores_list)

            for _ in range(len(proc_list)):
                proc_list, queue_list, scores_list = finish_proc(auc_list, dataset_name, f1_list, global_args,
                                                                 model_list, proc_list,
                                                                 queue_list, global_sampled_examples_list,
                                                                 f'round_{train_round}_single_model',
                                                                 most_anomalous_trn_examples_list, scores_list)

            if global_args.ensemble_method == 'isml' and len(scores_list) > 1:
                num_of_anomalies = int(y.sum())
                pred_vec, isml_pred_list = Trainer.pred_isml(num_of_anomalies, scores_list, y)
                ensemble_auc_list.append(sklearn.metrics.roc_auc_score(y, isml_pred_list))
                ensemble_f1_list.append(sklearn.metrics.f1_score(y, pred_vec))

            scores_list = []

        print(f'Run time: {time.time() - start}')

        if train_round + 1 < global_args.num_training_rounds:
            global_sampled_examples = generate_samples(model_list, global_data_premutations,
                                                       global_sampled_examples_list,
                                                       most_anomalous_trn_examples_list, n_samples,
                                                       global_args.use_most_ano_examples)

        if global_args.use_wandb:
            columns = [f"round_{train_round}_isml_F1", f"round_{train_round}_isml_AUC", f"round_{train_round}_F1",
                       f"round_{train_round}_AUC", f"round_{train_round}_f1_std", f"round_{train_round}_auc_std"]
            f1_mean = np.mean(f1_list)
            auc_mean = np.mean(auc_list)
            f1_std = np.std(f1_list)
            auc_std = np.std(auc_list)
            ensemble_auc_mean = np.mean(ensemble_auc_list)
            ensemble_f1_mean = np.mean(ensemble_f1_list)
            results = [ensemble_f1_mean, ensemble_auc_mean, f1_mean, auc_mean, f1_std, auc_std]

            for name, metric in zip(columns, results):
                mlflow.log_metric(f"{dataset_name}_{name}", metric)


def data_premutations(dataset_name, global_args):
    args = utils.get_args()
    ######HP#####
    for key, val in vars(global_args).items():
        args.__setattr__(key, val)

    data = get_preprocessed_data(args, dataset_name, 0)
    num_features = data.trn.x.shape[1]
    n_samples = data.trn.x.shape[0]
    return Trainer.get_data_perumutations(args.num_ensemble_models, num_features, args.permute_model_input), n_samples, data.tst.y


def finish_proc(auc_list, dataset_name, f1_list, global_args, model_list, proc_list, queue_list,
                global_sampled_examples_list, log_name, most_anomalous_trn_examples_list, scores_list):
    ensemble_auc_score, ensemble_f1_score, curr_model_list, past_seed,  curr_ensemble_sampled_examples, lr_list, \
        most_anomalous_trn_examples, scores_ll_list = queue_list[0].get()
    queue_list[0].task_done()

    # if global_args.use_wandb:
    #     mlflow.log_metric(f"{log_name}_f1_{dataset_name}", ensemble_f1_score, past_seed)

    # for iter, int_score in enumerate(interim_ensemble_f1_list):
    #     mlflow.log_metric(f"{log_name}_interim_f1_{dataset_name}_seed_{past_seed}", int_score, iter)

    # if global_args.ll_loss_lambda > 0:
    #     for iter, int_score in enumerate(interim_ensemble_ll_list):
    #         mlflow.log_metric(f"{log_name}_interim_ll_{dataset_name}_seed_{past_seed}", int_score, iter)
    #
    # if global_args.cdf_loss_lambda > 0:
    #     for iter, int_score in enumerate(interim_ensemble_cdf_list):
    #         mlflow.log_metric(f"{log_name}_interim_cdf_{dataset_name}_seed_{past_seed}", int_score, iter)

    # for iter, lr in enumerate(lr_list):
    #     mlflow.log_metric(f"LR_{dataset_name}_seed_{past_seed}", int_score, iter)

    global_sampled_examples_list.append(curr_ensemble_sampled_examples)
    most_anomalous_trn_examples_list.append(most_anomalous_trn_examples)
    auc_list.append(ensemble_auc_score)
    f1_list.append(ensemble_f1_score)
    model_list.append(curr_model_list)
    scores_list.append(scores_ll_list)
    queue_list = queue_list[1:]
    proc_list[0].join()
    proc_list = proc_list[1:]
    return proc_list, queue_list, scores_list


def generate_samples(ensemble_list, global_data_premutations, global_sampled_examples_list,
                     most_anomalous_trn_examples_list, n_samples, use_most_ano_examples):
    ensemble_ind_list = list(range(len(ensemble_list)))
    global_mean_ll_samples = [[] for _ in range(len(ensemble_list[0]))]
    global_sampled_examples = [[] for _ in range(len(ensemble_list[0]))]
    num_to_sample = int(np.max([np.min([n_samples / 50, 50]), 2]))
    for curr_ensemble_ind, emsemble in enumerate(ensemble_list):
        for curr_model_ind, model in enumerate(emsemble):
            curr_sampled_examples = global_sampled_examples_list[curr_ensemble_ind][curr_model_ind]
            global_sampled_examples[curr_model_ind].append(curr_sampled_examples)

            ll_samples_list = []
            for ensemble_ind in ensemble_ind_list:
                if ensemble_ind != curr_ensemble_ind:
                    ll, cdf = ensemble_list[ensemble_ind][curr_model_ind].forward_vec(curr_sampled_examples)
                    ll = ll.sum(dim=1)
                    ll_samples_list.append(ll)

            global_mean_ll_samples[curr_model_ind].append(torch.mean(torch.stack(ll_samples_list), dim=0))
    global_sampled_examples = [torch.cat(sampled_examples, dim=0) for sampled_examples in global_sampled_examples]

    best_samples_ind = [[] for _ in range(len(global_mean_ll_samples))]
    for model_ind, model_type_ll_list in enumerate(global_mean_ll_samples):
        for seed_ll_list in model_type_ll_list:
            best_samples_ind[model_ind].append(torch.argsort(seed_ll_list, descending=True)[-num_to_sample:])

        best_samples_ind[model_ind] = np.concatenate(best_samples_ind[model_ind], axis=0)

    for ind, sampled_ind in enumerate(best_samples_ind):
        global_sampled_examples[ind] = global_sampled_examples[ind][sampled_ind].detach().cpu().numpy()
        global_sampled_examples[ind] = global_sampled_examples[ind][:,
                                       invert_permutation(global_data_premutations[ind])]
    most_anomalous_trn_examples = np.concatenate(most_anomalous_trn_examples_list, axis=1).squeeze()
    global_sampled_examples = np.concatenate(global_sampled_examples, axis=0)
    if use_most_ano_examples:
        global_sampled_examples = np.concatenate([global_sampled_examples, most_anomalous_trn_examples], axis=0)
    return global_sampled_examples


def main():
    global_args.device = global_args.device.split(',')
    if global_args.use_wandb:
        mlflow.set_tracking_uri(uri=f'mlruns')
        exp = mlflow.get_experiment_by_name(name='NITSA')
        if not exp:
            experiment_name = mlflow.create_experiment(name='NITSA')
        else:
            experiment_name = exp.experiment_id
        mlflow.set_experiment(experiment_id=experiment_name)
        wandb_run_name = f"n_ensemble_{global_args.num_ensemble_models}_cascade_{global_args.cascade_training}_" \
                         f"num_seeds_{global_args.num_seeds}_batch_size_{global_args.batch_size}_factor_" \
                         f"{global_args.variable_batch_size_factor}_lr_" \
                         f"{global_args.learning_rate}_dropout_{global_args.dropout}_epochs_{global_args.epochs}_" \
                         f"pre_train_epochs_{global_args.pre_train_epochs}_add_res_{global_args.add_residual_connections}_opt_{global_args.optimizer}_" \
                         f"ensemble_method_{global_args.ensemble_method}_ll_lambda_{global_args.ll_loss_lambda}_" \
                         f"cdf_lambda_{global_args.cdf_loss_lambda}_hidden_dim_{global_args.hidden_dim}_use_most_ano_examples_{global_args.use_most_ano_examples}"
        mlflow.start_run(experiment_id=experiment_name, run_name=wandb_run_name)
        mlflow.log_param('run_name', wandb_run_name)

    datasets = []
    for file in glob.glob(f"{global_args.data_dir}/*mat"):
        datasets.append(file.split("/")[-1][:-4])

    for file in glob.glob(f"{global_args.data_dir}/*csv"):
        datasets.append(file.split("/")[-1])

    for file in glob.glob(f"{global_args.data_dir}/*npz"):
        datasets.append(file.split("/")[-1])

    for file in glob.glob(f"{global_args.data_dir}/*arff"):
        datasets.append(file.split("/")[-1])

    for dataset_name in tqdm(datasets):
        if len(global_args.single_dataset_name) > 0 and dataset_name != global_args.single_dataset_name:
            continue

        print(dataset_name)
        try:
            run_one_ds_many_seeds(dataset_name, global_args)
        except Exception as e:
            print(e)

    if global_args.use_wandb:
        mlflow.end_run()


if __name__ == "__main__":
    global global_args
    global_args = get_args()
    main()
