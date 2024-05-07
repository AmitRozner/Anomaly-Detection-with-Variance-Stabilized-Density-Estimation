import os
import torch
import argparse
from scipy import io
from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

def list_str_to_list(s):
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')
    s = [int(x) for x in s]
    return s

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='cardio')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=500)
    parser.add_argument('--min_batch_size', type=int, default=16)
    parser.add_argument('-hi', '--hidden_dim', type=int, default=512)
    parser.add_argument('-nr', '--n_residual_blocks', type=int, default=2)
    parser.add_argument('-n', '--patience', type=int, default=2)
    parser.add_argument('-ga', '--gamma', type=float, default=1)
    parser.add_argument('-pd', '--polyak_decay', type=float, default=0.995)
    parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,2]')
    parser.add_argument('-r', '--rotate', action='store_true')
    parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-p', '--dropout', type=float, default=-1.0)
    parser.add_argument('-rc', '--add_residual_connections', action="store_true")
    parser.add_argument('-bm', '--bound_multiplier', type=float, default=1.0)
    parser.add_argument('--train_test_normal_split', type=float, default=0.5)
    parser.add_argument('-pe', '--permute_data', action='store_true')
    parser.add_argument('--data_dir', type=str, default="./datasets")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pre_train_epochs', type=int, default=0)
    parser.add_argument('--normalize_dataset', type=str, default='zscore')
    parser.add_argument('--num_ensemble_models', type=int, default=1)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--user', type=str, default='amit')
    parser.add_argument('--use_mp', action="store_true")
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--cascade_training', action="store_true")
    parser.add_argument('--use_most_ano_examples', action="store_true")
    parser.add_argument('--single_dataset_name', type=str, default='')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--scheduler', type=str, choices=['one_cycle', "cos_annealing"], default='cos_annealing')
    parser.add_argument('--variable_batch_size_factor', type=int, default=1)
    parser.add_argument('--num_parallel_procs', type=int, default=2)
    parser.add_argument('--ensemble_method', type=str, default='majority', choices=['majority', 'isml'])
    parser.add_argument('--cdf_loss_lambda', type=float, default=0.0)
    parser.add_argument('--ll_loss_lambda', type=float, default=1.0)
    parser.add_argument('--var_loss_lambda', type=float, default=3.33333333)
    parser.add_argument('--recon_loss_lambda', type=float, default=0.0)
    parser.add_argument('--ll_recon_loss_lambda', type=float, default=0.0)
    parser.add_argument('--early_stopping', action="store_true")
    parser.add_argument('--num_training_rounds', type=int, default=2)
    parser.add_argument('--scarf', action="store_true")
    parser.add_argument('--num_reduce_dims', nargs='+', type=float, default=[])
    parser.add_argument('--num_reduce_dims_pca', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--permute_model_input', action="store_false")
    args = parser.parse_args()
    return args


def build_tridiagonal(n=10000, d=30, k=1, permute=False):
    precov = np.random.normal(size=(d, d))
    precov = np.matmul(precov, precov.T)
    cov = np.tril(np.triu(precov, -k), k)
    cov = cov / np.diag(cov).mean()

    pre_x = np.random.normal(size=(n, d))
    x = np.matmul(pre_x, cov)

    # normalize
    m = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    assert m.shape == x[0:1].shape and std.shape == x[0:1].shape
    x = (x - m) / std
    return Dataset(x.astype(np.float))


def permute_data(dataset):
    d = dataset.trn.x.shape[1]
    train_idx = len(dataset.trn.x)
    val_idx = train_idx + len(dataset.val.x)
    x = np.concatenate([dataset.trn.x, dataset.val.x, dataset.tst.x], axis=0)

    P = np.eye(d)
    P = P[np.random.permutation(d)]
    permuted_x = np.matmul(x, P)
    assert np.allclose(np.matmul(permuted_x, P.T), x)

    return Dataset(permuted_x.astype(np.float), train_idx=train_idx, val_idx=val_idx), P.astype(np.float)


def build_double_tridiagonal(n=2000, d=30, k=5, permute=False):
    precov = np.random.normal(size=(d, d))
    precov = np.matmul(precov, precov.T)
    cov = np.tril(np.triu(precov, -k), k)
    cov = cov / np.diag(cov).mean()

    pre_x1 = np.random.normal(size=(n, d))
    x1 = np.matmul(pre_x1, cov)

    # normalize
    m1 = np.mean(x1, axis=0, keepdims=True)
    std1 = np.std(x1, axis=0, keepdims=True)
    assert m1.shape == x1[0:1].shape and std1.shape == x1[0:1].shape
    x1 = (x1 - m1) / std1

    pre_x2 = np.random.normal(size=(n, d))
    x2 = np.matmul(pre_x2, cov)

    # normalize
    m2 = np.mean(x2, axis=0, keepdims=True)
    std2 = np.std(x2, axis=0, keepdims=True)
    assert m2.shape == x2[0:1].shape and std2.shape == x2[0:1].shape
    x2 = (x2 - m2) / std2
    x = np.hstack((x1, x2))
    return Dataset(x.astype(np.float))


class Dataset:
    def __init__(self, args, x, y, permute=False, train_idx=0, val_idx=0, seed=0):
        # splits x into train, val, and test
        self.n = len(x)
        if permute:
            p = np.random.permutation(self.n)
            x = x[p]

        class DataHolder:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        idx_norm = y == 0
        idx_out = y == 1
        x_train, x_test_reg, y_train, y_test_reg = train_test_split(x[idx_norm.squeeze()], y[idx_norm.squeeze()],
                                                                    test_size=args.train_test_normal_split,
                                                                    random_state=0)
        x_test_anomaly, y_test_anomaly = x[idx_out.squeeze()], y[idx_out.squeeze()],

        x_test = np.concatenate((x_test_reg, x_test_anomaly), axis=0)
        y_test = np.concatenate((y_test_reg, y_test_anomaly), axis=0)

        self.trn = DataHolder(x_train, y_train)
        self.val = DataHolder(x[0:0], y[0:0])
        self.tst = DataHolder(x_test, y_test)


def create_batcher(x, batch_size=1, device='cpu', phase='train'):
    idx = 0

    if phase == 'train':
        p = torch.randperm(len(x))
        x = x[p]

    while idx + batch_size < len(x):
        yield torch.tensor(x[idx:idx + batch_size], device=device).float()
        idx += batch_size
    else:
        yield torch.tensor(x[idx:], device=device).float()


def load_dataset(args, dataset, seed=0):
    if dataset == 'gas':
        # training set size: 852,174
        data = gas.GAS()
        default_dropout = 0.1
    elif dataset == 'power':
        # training set size: 1,659,917
        data = power.POWER()
        default_dropout = 0.1
    elif dataset == 'miniboone':
        # training set size: 29,556
        data = miniboone.MINIBOONE()
        default_dropout = 0.3
    elif dataset == 'hepmass':
        # training set size: 315,123
        data = hepmass.HEPMASS()
        default_dropout = 0.5
        default_pateince = 3
    elif dataset == 'bsds300':
        # training set size: 1,000,000
        data = bsds300.BSDS300()
        default_dropout = 0.2
    elif dataset == 'tridiagonal':
        data = build_double_tridiagonal()
        default_dropout = 0.0
    elif dataset == 'http':
        import mat73
        d_in = mat73.loadmat(os.path.join(args.data_dir, dataset + '.mat'))
        data = Dataset(args, d_in['X'].astype(np.float), d_in['y'].astype(np.float), seed=seed)
        default_dropout = 0.0
    elif dataset == 'abalone':
        data = pd.read_csv(os.path.join(args.data_dir, dataset + '.data'), header=None, sep=',')
        data = data.rename(columns={8: 'y'})
        data['y'].replace([8, 9, 10], -1, inplace=True)
        data['y'].replace([3, 21], 0, inplace=True)
        data.iloc[:, 0].replace('M', 0, inplace=True)
        data.iloc[:, 0].replace('F', 1, inplace=True)
        data.iloc[:, 0].replace('I', 2, inplace=True)
        test = data[data['y'] == 0]
        normal = data[data['y'] == -1].sample(frac=1)
        test.loc[:, 'y'] = test.loc[:, 'y'].replace(0, 1)
        normal.loc[:, 'y'] = normal.loc[:, 'y'].replace(-1, 0)
        d_in = np.concatenate([normal, test], axis=0)
        data = Dataset(args, d_in[:, :-1], np.expand_dims(d_in[:, -1], -1), seed=seed)
        return data, 1
    elif dataset == 'ecoli':
        dataset = pd.read_csv(os.path.join(args.data_dir, dataset +  '.data'), header=None, sep='\s+')
        dataset = dataset.iloc[:, 1:]
        anomalies = np.array(
            dataset[(dataset.iloc[:, 7] == 'omL') | (dataset.iloc[:, 7] == 'imL') | (dataset.iloc[:, 7] == 'imS')])[:,
                    :-1]
        normals = np.array(dataset[(dataset.iloc[:, 7] == 'cp') | (dataset.iloc[:, 7] == 'im') | (
                    dataset.iloc[:, 7] == 'pp') | (dataset.iloc[:, 7] == 'imU') | (dataset.iloc[:, 7] == 'om')])[:, :-1]
        normals = normals.astype('double')
        anomalies = anomalies.astype('double')
        normals = np.concatenate((normals, np.zeros([normals.shape[0], 1]).astype(np.float32)), axis=1)
        anomalies = np.concatenate((anomalies, np.ones([anomalies.shape[0], 1]).astype(np.float32)), axis=1)
        d_in = np.concatenate([normals, anomalies], axis=0)
        data = Dataset(args, d_in[:, :-1], np.expand_dims(d_in[:, -1], -1), seed=seed)
        return data, 1
    elif dataset.endswith('csv'):
        # loading data
        df = pd.read_csv(os.path.join(args.data_dir, dataset))
        labels = df['class']
        x_df = df.drop(['class'], axis=1)
        x = x_df.values

        data = Dataset(args, x.astype(np.float), labels.values.astype(np.float), seed=seed)
        return data, 1
    elif dataset.endswith("npz"):
        data = np.load(os.path.join(args.data_dir, dataset), allow_pickle=True)
        data = Dataset(args, data['X'].astype(np.float), data['y'].astype(np.float), seed=seed)
        return data, 1
    elif dataset.endswith("arff"):
        dataset_name = dataset
        dataset, meta = arff.loadarff(os.path.join(args.data_dir, dataset))
        dataset = pd.DataFrame(dataset)
        classes = dataset.iloc[:, -1]
        dataset = dataset.iloc[:, :-1]
        dataset = pd.get_dummies(dataset.iloc[:, :-1])
        dataset = pd.concat((dataset, classes), axis=1)
        if "seismic" in dataset_name:
            normals = dataset[dataset.iloc[:, -1] == b'0'].values
            anomalies = dataset[dataset.iloc[:, -1] == b'1'].values
        elif "mulcross" in dataset_name:
            normals = dataset[dataset.iloc[:, -1] == b'Normal'].values
            anomalies = dataset[dataset.iloc[:, -1] == b'Anomaly'].values
        else:
            raise ValueError
        normals = torch.tensor(normals[:, :-1].astype('float32'))
        anomalies = torch.tensor(anomalies[:, :-1].astype('float32'))
        X = np.asarray(torch.cat((normals, anomalies), dim=0))
        Y = np.asarray(torch.cat((torch.zeros(normals.shape[0]).view(-1, 1), torch.ones(anomalies.shape[0]).view(-1, 1)), dim=0))
        data = Dataset(args, X.astype(np.float), Y.astype(np.float), seed=seed)
        return data, 1
    else:  # if dataset == 'cardio':
        d_in = io.loadmat(os.path.join(args.data_dir, dataset + '.mat'))
        data = Dataset(args, d_in['X'].astype(np.float), d_in['y'].astype(np.float), seed=seed)
        # Y = d_in['y']
        default_dropout = 0.0
    return data, d_in


def compute_ap_auc(actuals, predictions):
    # compute ap and auc
    # -1 outlier, 1 inlier
    # 0 inlier, 1 outlier
    # print(actuals)
    # print(predictions)
    # print(actuals.shape)
    map = average_precision_score(actuals, predictions)
    aucs = roc_auc_score(actuals, predictions)
    # torch.where(predictions>0.5)
    # acc = balanced_accuracy_score(actuals, predictions)
    # acc = accuracy_score(actuals_one_hot, predictions)
    # print(acc)
    acc = 0
    # map = np.mean(aps)
    # mauc = np.mean(aucs)
    # return map, mauc
    return map, aucs

def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def normalize_dataset(data, normalize_dataset):
    trn_mean = np.mean(data.trn.x, axis=0)
    trn_std = np.std(data.trn.x, axis=0)

    if np.any(trn_std < 1e-7):
        # Remove any column which has std of zero
        col_to_keep = trn_std >= 1e-7
        trn_mean = trn_mean[col_to_keep]
        trn_std = trn_std[col_to_keep]
        data.trn.x = data.trn.x[:, col_to_keep]
        data.tst.x = data.tst.x[:, col_to_keep]
        data.val.x = data.val.x[:, col_to_keep]

    if normalize_dataset == 'zscore':
        data.trn.x = (data.trn.x - trn_mean) / trn_std
        data.tst.x = (data.tst.x - trn_mean) / trn_std
        data.val.x = (data.val.x - trn_mean) / trn_std

    if normalize_dataset == 'minmax':
        trn_max = np.max(np.abs(data.trn.x), axis=0)
        trn_min = np.min(np.abs(data.trn.x), axis=0)
        data.trn.x = (data.trn.x - trn_min) / (trn_max - trn_min)
        data.val.x = (data.val.x - trn_min) / (trn_max - trn_min)
        data.tst.x = (data.tst.x - trn_min) / (trn_max - trn_min)

    return data


def get_preprocessed_data(args, dataset_name, seed):
    data, _ = load_dataset(args, dataset_name, seed)
    data = normalize_dataset(data, args.normalize_dataset)

    if args.num_reduce_dims_pca > 0:
        pca = PCA(n_components=args.num_reduce_dims_pca)
        temp_data = pca.fit_transform(np.concatenate([data.trn.x, data.tst.x]))
        data.trn.x = temp_data[:len(data.trn.x), :]
        data.tst.x = temp_data[len(data.trn.x):, :]
    return data