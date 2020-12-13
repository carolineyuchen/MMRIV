import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import random
import scipy.sparse as sp
import matplotlib as mpl
import statistics
from itertools import product
import itertools
import math
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.linalg as la
from decimal import *
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise
import argparse
import os

parser = argparse.ArgumentParser(description='settings for simulation')

N = [500, 5000, 10000, 100000, 500000]
a_AY = 0.5
a_AZ = 0.5

m_e = [5, 0, 0, -1, 2]
C = [1, 1, 1, 1, 1]

parser.add_argument('--sem', type=str, help='sets SEM')
parser.add_argument('--fun-ay', type=str, help='sets function from a to y')

args = parser.parse_args()


def asina(a, b):
    return np.sin(b * a) * a


def expa(a, b):
    return np.exp(b * a)


func_dict = {'asina': asina, 'expa': expa}


def gen_eval_samples(test_sample_size, w_sample_thresh, axz, axzwy):
    inp = input('Check the input order is a, x, z. ans: y/n')
    if inp == 'y':
        pass
    else:
        raise ValueError('incorrectly ordered input.')

    assert axz.shape[0] == 3
    axz = axz[:, :test_sample_size]
    assert axz.shape[1] == test_sample_size

    assert axzwy.shape[0] == 5
    # assert axzwy.shape[1] == test_sample_size * 1000

    axz_out, y_av_out, w_samples_out, y_samples_out = [], [], [], []

    for i in range(test_sample_size):
        axz_all = axzwy[:3,:]
        axz_diff = axz_all - axz[:, i:i+1]
        print('axz_all: ', axz_all, '\n', 'axz vec: ', axz[:,i:i+1], '\n', 'difference: ', axz_diff)
        axz_valid_idx = (axz_diff > -0.12) * (axz_diff < 0.12)
        axz_valid_col_idx = np.prod(axz_valid_idx, axis=0)
        print('valid row idx: ', axz_valid_col_idx)
        num_valid = np.sum(axz_valid_col_idx)
        print('num valid: ', num_valid)
        if num_valid < w_sample_thresh:
            continue
        else:
            axz_valid_col_idx = np.nonzero(axz_valid_col_idx)
            print('valid indices: ', axz_valid_col_idx)
            subTuple = np.squeeze(axzwy[:, axz_valid_col_idx])
            subTuple = subTuple[:, :w_sample_thresh]
            y_axz_av = np.mean(subTuple[-1, :])
            y_samples_out.append(subTuple[-1, :])
            axz_out.append(axz[:, i])
            y_av_out.append(y_axz_av)
            w_samples_out.append(subTuple[-2,:])
            print('subTuples: ', subTuple)
            print('axzwy: ', axzwy)
            print('w_samples: ', subTuple[-2, :])
    axz_np = np.array(axz_out)
    y_np = np.array(y_av_out)
    axzy_np = np.concatenate([axz_np, y_np.reshape(-1,1)], axis=1)
    w_samples_out_np = np.array(w_samples_out)
    y_samples_out_np = np.array(y_samples_out)
    print('num eval tuples: ', axzy_np.shape[0], 'axzy: ', axzy_np, 'w_samples: ', w_samples_out_np)

    return axzy_np, w_samples_out_np, y_samples_out_np


def main(args):
    np.random.seed(100)
    U = np.random.chisquare(m_e[0], N[-1]).round(3)
    train_u, test_u, dev_u, rest_u = U[:3000], U[3000:4000], U[4000:5000], U[5000:]
    U_inst = np.ones(N[-1]).round(3)

    X = np.random.chisquare(m_e[0], N[-1]).round(3)  # generates 5000 X's to 3.d.p.
    train_x, test_x, dev_x, rest_x = X[:3000], X[3000:4000], X[4000:5000], X[5000:]
    X_inst = np.ones(N[-1]).round(3)

    # Z is noisy reading of U
    eZ = np.random.normal(m_e[3], C[3], N[-1])  # noise for Z
    Z = (eZ - U).round(3)
    train_z, test_z, dev_z, rest_z = Z[:3000], Z[3000:4000], Z[4000:5000], Z[5000:]
    Z_conU = (eZ - U_inst).round(3)  # constant U

    # noise for W
    eW = np.random.normal(m_e[4], C[4], N[-1])
    W = (eW + 2 * U).round(3)
    train_w, test_w, dev_w, rest_w = W[:3000], W[3000:4000], W[4000:5000], W[5000:]
    W_conU = (eW + 2 * U_inst).round(3)

    eA = np.random.normal(m_e[1], C[1], N[-1])
    A = (eA + 2 * U ** 0.5).round(3)
    train_a, test_a, dev_a, rest_a = A[:3000], A[3000:4000], A[4000:5000], A[5000:]
    A_conU = (eA + a_AZ * Z_conU + 2 * U_inst ** 0.5).round(3)

    eY = np.random.normal(m_e[2], C[2], N[-1])
    fun_ay = func_dict[args.fun_ay]
    Y = (fun_ay(a=A, b=a_AY) + eY - np.log10(U)).round(3)
    train_y, test_y, dev_y, rest_y = Y[:3000], Y[3000:4000], Y[4000:5000], Y[5000:]
    Y_conU = (fun_ay(a=A_conU, b=a_AY) + eY - np.log10(U_inst)).round(3)

    # causal ground truth
    do_A = np.linspace(1, 20, 20)
    EY_do_A = []
    for a in do_A:
        A_ = np.repeat(a, [N[1]])
        Y_do_A = (fun_ay(A_, b=a_AY) + eY[:N[1]] - np.log10(U[:N[1]])).round(3)
        eY_do_A = np.mean(Y_do_A)
        EY_do_A.append(eY_do_A)

    EY_do_A = np.array(EY_do_A)

    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/main_{}.npz'.format(args.sem)),
             splits=['train', 'test', 'dev'],
             train_y=train_y,
             train_a=train_a,
             train_z=train_z,
             train_w=train_w,
             train_u=train_u,
             test_y = test_y,
             test_a = test_a,
             test_z = test_z,
             test_w = test_w,
             test_u = test_u,
             dev_y = dev_y,
             dev_a = dev_a,
             dev_z = dev_z,
             dev_w = dev_w,
             dev_u = dev_u)

    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/do_A_{}.npz'.format(args.sem)),
             do_A = do_A,
             gt_EY_do_A = EY_do_A)

    # plotting
    D = pd.DataFrame([U[:300], A[:300], Y[:300], Z[:300], W[:300]]).T
    D.columns = ['U', 'A', 'Y','Z', 'W']
    O = pd.DataFrame([A[:300], Y[:300], Z[:300], W[:300]]).T
    O.columns = ['A', 'Y', 'Z', 'W']
    D_conU = pd.DataFrame([U_inst[:300], A_conU[:300], Y_conU[:300], Z_conU[:300], W_conU[:300]]).T
    D_conU.columns = ['U', 'A', 'Y', 'Z', 'W']

    ecorr_v = D.corr()
    ecorr_v.columns = ['U', 'A', 'Y', 'Z', 'W']
    ecorr_O = O.corr()
    ecorr_O.columns = ['A', 'Y', 'Z', 'W']
    ecorr_v_conU = D_conU.corr()
    ecorr_v_conU.columns = ['U', 'A', 'Y', 'Z', 'W']


    sem = args.sem
    if not os.path.exists(PATH+sem):
        os.mkdir(PATH + sem)
    for v in ['U', 'A', 'Y', 'Z', 'W']:
        sns.displot(D, x=v, label=v, kde=True), plt.savefig(PATH + sem + '/' + v + '_dist.png'), plt.close()

    sns.set_theme(font="tahoma", font_scale=1)
    sns.pairplot(D), plt.savefig(PATH + sem + '/' + 'full_pairwise.png'), plt.close()
    sns.pairplot(O), plt.savefig(PATH + sem + '/' + 'observed_pairwise.png'), plt.close()
    sns.pairplot(D_conU), plt.savefig(PATH + sem + '/' + 'fixed_U_pairwise.png'), plt.close()

    sns.heatmap(ecorr_v, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_all.png'), plt.close()
    sns.heatmap(ecorr_O, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_observed.png'), plt.close()
    sns.heatmap(ecorr_v_conU, annot=True, fmt=".2"), plt.savefig(PATH + sem + '/' + 'corr_fixed_U.png'), plt.close()

    # generating conditional expectation
    print('expectation evaluation starts.')
    test_sample_sz = 1000
    w_sample_thresh = 20
    axz = np.vstack([test_a, test_x, test_z])  # shape: 3 x 1000
    axzwy = np.vstack([rest_a, rest_x, rest_z, rest_w, rest_y])
    axzy_np, w_samples_out_np, y_samples_out_np = gen_eval_samples(test_sample_size=test_sample_sz,
                                                                   w_sample_thresh=w_sample_thresh,
                                                                   axz=axz,
                                                                   axzwy=axzwy)
    np.savez(os.path.join(PATH, '../data/zoo/sim_1d_no_x/cond_exp_metric_{}.npz'.format(args.sem)),
             axzy=axzy_np,
             w_samples=w_samples_out_np,
             y_samples=y_samples_out_np)


if __name__ == '__main__':
    main(args)