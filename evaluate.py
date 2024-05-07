import os
import numpy as np
import sklearn
import torch
from utils import get_args, get_preprocessed_data
from trainer import Trainer
from scipy import stats

if __name__ == "__main__":
    VERBOSE = False
    args = get_args()
    all_results = {}
    args.data_dir = "./Classical"
    # args.save_path = "results_no_scarf"
    args.save_path = "ensemble_5"
    # args.save_path = "ours_var_0.1"
    for root, dirs, files in os.walk(args.save_path):
        for name in files:
            # Number of permutations ablation
            # if name.endswith("_num_4") or name.endswith("_num_3") or name.endswith("_num_2") or name.endswith("_num_1"):
            #     continue
            if name.startswith('raw_ll'):
                curr_ds_name = (name.split('raw_ll_')[1]).split('_seed')[0]

                curr_seed = name.split('seed_')[1][0]
                if curr_ds_name not in all_results.keys():
                    all_results[curr_ds_name] = {}
                    all_results[curr_ds_name]['raw_ll'] = {}
                    all_results[curr_ds_name]['recon'] = {}

                if curr_seed not in all_results[curr_ds_name]['raw_ll'].keys():
                    all_results[curr_ds_name]['raw_ll'][curr_seed] = []
                    all_results[curr_ds_name]['recon'][curr_seed] = []

                all_results[curr_ds_name]['raw_ll'][curr_seed].append(torch.load(os.path.join(root, name)))
                recon_name = name.replace('raw_ll', 'recon')
                if os.path.exists(os.path.join(root, recon_name)):
                    all_results[curr_ds_name]['recon'][curr_seed].append(torch.load(os.path.join(root, recon_name)))
                else:
                    all_results[curr_ds_name]['recon'][curr_seed].append([])

    data_diff_mean_var = []
    for dataset_name in all_results.keys():
        curr_ds_results = all_results[dataset_name]
        data = get_preprocessed_data(args, dataset_name, 0)
        num_of_anomalies = int(data.tst.y.sum())
        ensemble_auc_list = []
        for seed in curr_ds_results['raw_ll'].keys():
            seed_ll_scores = []
            curr_ensemble_score = []
            for result_ll, recon in zip(curr_ds_results['raw_ll'][seed], curr_ds_results['recon'][seed]):
                if len(recon):
                    full_results = np.concatenate([result_ll, -recon], axis=1)
                else:
                    full_results = result_ll
                    anomaly_mat = result_ll[-num_of_anomalies:]
                    normal_mat = result_ll[-2*num_of_anomalies:-num_of_anomalies]
                    anomaly_mean_var = np.mean(np.var(anomaly_mat, axis=0))
                    normal_mean_var = np.mean(np.var(normal_mat, axis=0))
                    data_diff_mean_var.append([dataset_name, anomaly_mean_var, normal_mean_var])
                # continue
                cols_to_keep = np.std(full_results, axis=0) >= 1e-3
                full_results = full_results[:, cols_to_keep]
                result_zs = stats.zscore(full_results, axis=0)
                ll_new_t = full_results
                ll_new = np.mean(ll_new_t, axis=1)
                if len(data.trn.y) < len(ll_new):
                    ll_new = ll_new[len(data.trn.y):]

                seed_ll_scores.append(-ll_new)
                curr_ensemble_score.append(sklearn.metrics.roc_auc_score(data.tst.y, seed_ll_scores[-1]))
                if VERBOSE:
                    print(f'{dataset_name}_{seed}_{curr_ensemble_score[-1]}')
            # continue
            if len(seed_ll_scores) > 1:
                 pred_vec, isml_pred_list = Trainer.pred_isml(num_of_anomalies, seed_ll_scores, data.tst.y)
            else:
                isml_pred_list = seed_ll_scores[-1]

            if 0: #Calculate mean ensemble ablation
                ensemble_auc_list.append(np.mean(curr_ensemble_score))
            else:
                ensemble_auc_list.append(sklearn.metrics.roc_auc_score(data.tst.y, isml_pred_list))
                # ensemble_auc_list.append(sklearn.metrics.f1_score(data.tst.y, pred_vec))
            if VERBOSE:
                print(f'ISML_{dataset_name}_{seed}_{ensemble_auc_list[-1]}')
        # print(f'Mean ISML F1{dataset_name}: {np.round(100 * np.mean(ensemble_auc_list), 1)}')
        print(f'Mean ISML {dataset_name}: {np.round(100*np.mean(ensemble_auc_list),1)}+{np.round(100*np.std(ensemble_auc_list),1)}')

    import pandas as pd
    from matplotlib import pyplot as plt
    pd_data_dmv = pd.DataFrame(data_diff_mean_var).groupby(0).mean().sort_values(1, ascending=True)
    labels = list(pd_data_dmv.index)
    for ind in range(len(labels)):
        if labels[ind].startswith("creditcard"):
            labels[ind] = "Fraud"

        split_ = labels[ind].split('-')
        if len(split_) > 1:
            labels[ind] = ''.join(split_[0])

        split_ = labels[ind].split('_')
        if len(split_) > 1:
            labels[ind] = ''.join(split_[1:])

        split_ = labels[ind].split('.')
        if len(split_) > 1:
            labels[ind] = ''.join(split_[:-1])

        split_ = labels[ind].split('_')
        if len(split_) > 1:
            labels[ind] = ''.join(split_[1:])

        if labels[ind] == "NB15traintestbackdoor":
            labels[ind] = "Backdoor"

        if labels[ind] == "phpGGVhl9":
            labels[ind] = "Mulcross"

        if labels[ind] == "baldvsnonbaldnormalised":
            labels[ind] = "celeba"

        if labels[ind] == "donors10featnomissingnormalised":
            labels[ind] = "Donors"

        labels[ind] = labels[ind].capitalize()

    fig = plt.figure(figsize=(18, 10))
    y_values = pd_data_dmv[1].values
    above_1 = np.sum(y_values > 1)
    color = ['red'] * (len(y_values) - above_1) + ['green'] * above_1
    plt.bar(labels, np.array(pd_data_dmv[1].values), color=color, width=0.8)
    plt.yscale("log")
    plt.axhline(y=1, color='k', linestyle='--')
    # plt.ylim([-10, 2])
    fig.autofmt_xdate()
    plt.xlabel('Dataset name', fontsize=18)
    plt.ylabel(r'$\sigma^2(ll_{anomalous})/\sigma^2(ll_{normal})$', fontsize=18)
    plt.show()
    # plt.savefig('meanvarll.png', bbox_inches='tight')
x = np.arange(len(pd_data_dmv[1].values))  # the label locations
width = 0.4  # the width of the bars
multiplier = 0
fig, ax = plt.subplots(layout='constrained')
y_values = {'Normal': pd_data_dmv[2].values, 'Anomalous': pd_data_dmv[1].values}
for attribute, measurement in y_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1
plt.yscale("log")
fig.autofmt_xdate()
plt.xlabel('Dataset name', fontsize=18)
ax.set_xticks(x + width, labels)
ax.legend(loc='upper left', ncols=2)
fig.set_figwidth(16)
fig.set_figheight(10)
plt.ylabel(r'$\sigma^2(log likelihood)$', fontsize=18)
plt.show()