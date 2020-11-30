import os,sys,torch,add_path
import torch.autograd as ag
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scenarios.abstract_scenario import AbstractScenario
from early_stopping import EarlyStopping
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import scipy
from joblib import Parallel, delayed
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH,_sqdist,FCNN, CNN, bundle_az_aw, visualise_ATEs





def run_experiment_nn(sname,datasize,indices=[],seed=527,training=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if len(indices)==2:
        lr_id, dw_id = indices
    elif len(indices)==3:
        lr_id, dw_id,W_id = indices
    # load data
    folder = ROOT_PATH+"/MMR_IVs/results/zoo/" + sname + "/"
    os.makedirs(folder, exist_ok=True)
    
    train, dev, test = load_data(ROOT_PATH+"/data/zoo/"+sname+'/main.npz', Torch=True)
    Y = torch.cat((train.y, dev.y), dim=0).float()
    AZ_train, AW_train = bundle_az_aw(train.a, train.z, train.w, Torch=True)
    AZ_test, AW_test = bundle_az_aw(test.a, test.z, test.w, Torch=True)
    AZ_dev, AW_dev = bundle_az_aw(dev.a, dev.z, test.w, Torch=True)

    X, Z= torch.cat((AW_train,AW_dev),dim=0).float(), torch.cat((AZ_train, AZ_dev),dim=0).float()
    test_X, test_Y = AW_test.float(),test.y.float()  # TODO: is test.g just test.y?
    n_train = train.a.shape[0]
    # training settings
    n_epochs = 1000
    batch_size = 1000 if train.a.shape[0]>1000 else train.a.shape[0]

    # load expectation eval data
    axzy = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric.npz')['axzy']
    w_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric.npz')['w_samples']
    y_samples = np.load(ROOT_PATH + "/data/zoo/" + sname + '/cond_exp_metric.npz')['y_samples']
    y_axz = axzy[:, -1]
    ax = axzy[:, :2]

    # kernel
    kernel = Kernel('rbf', Torch=True)
    a = get_median_inter_mnist(AZ_train)
    a = torch.tensor(a).float()
    # training loop
    lrs = [2e-4,1e-4,5e-5] # [3,5]
    decay_weights = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6] # [11,5]

    def my_loss(output, target, indices, K):
        d = output - target
        if indices is None:
            W = K
        else:
            W = K[indices[:, None], indices]
            # print((kernel(Z[indices],None,a,1)+kernel(Z[indices],None,a/10,1)+kernel(Z[indices],None,a*10,1))/3-W)
        loss = d.T @ W @ d / (d.shape[0]) ** 2
        return loss[0, 0]

    def conditional_expected_loss(net, ax, w_samples, y_samples, y_axz, x_on):
        if not x_on:
            ax = ax[:, 0:1]
        num_reps = w_samples.shape[1]
        assert len(ax.shape) == 2
        assert ax.shape[1] < 3
        assert ax.shape[0] == w_samples.shape[0]
        print('number of points: ', w_samples.shape[0])

        ax_rep = np.repeat(ax, [num_reps], axis=0)
        assert ax_rep.shape[0] == (w_samples.shape[1] * ax.shape[0])

        w_samples_flat = w_samples.flatten().reshape(-1,1)
        nn_inp_np = np.concatenate([ax_rep, w_samples_flat], axis=-1)
        # print('nn_inp shape: ', nn_inp_np.shape)
        nn_inp = torch.as_tensor(nn_inp_np).float()
        nn_out = net(nn_inp).detach().cpu().numpy()
        nn_out = nn_out.reshape([-1, w_samples.shape[1]])
        y_axz_recon = np.mean(nn_out, axis=1)
        assert y_axz_recon.shape[0] == y_axz.shape[0]
        mean_abs_error = np.mean(np.abs(y_axz - y_axz_recon))

        # for debugging compute the mse between y samples and h
        y_samples_flat = y_samples.flatten()
        mse = np.mean((y_samples_flat - nn_out.flatten())**2)

        return mean_abs_error, mse

    def fit(x,y,z,dev_x,dev_y,dev_z,a,lr,decay_weight, ax, y_axz, w_samples, n_epochs=n_epochs):
        if 'mnist' in sname:
            train_K = torch.eye(x.shape[0])
        else:
            train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
        if dev_z is not None:
            if 'mnist' in sname:
                dev_K = torch.eye(x.shape[0])
            else:
                dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3
        n_data = x.shape[0]
        net = FCNN(x.shape[1]) if sname not in ['mnist_x', 'mnist_xz'] else CNN()
        es = EarlyStopping(patience=10)  # 10 for small
        optimizer = optim.Adam(list(net.parameters()), lr=lr, weight_decay=decay_weight)

        test_errs, dev_errs, exp_errs, mse_s = [], [], [], []

        for epoch in range(n_epochs):
            permutation = torch.randperm(n_data)

            for i in range(0, n_data, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = x[indices], y[indices]

                # training loop
                def closure():
                    optimizer.zero_grad()
                    pred_y = net(batch_x)
                    loss = my_loss(pred_y, batch_y, indices, train_K)
                    loss.backward()
                    return loss

                optimizer.step(closure)  # Does the update
            if epoch % 5 == 0 and epoch >= 50 and dev_x is not None:  # 5, 10 for small # 5,50 for large
                g_pred = net(test_X)  # TODO: is it supposed to be test_X here? A: yes I think so.
                test_err = ((g_pred-test_Y)**2).mean() # TODO: why isn't this loss reweighted? A: because it is supposed to measure the agreement between prediction and labels.
                if epoch == 50 and 'mnist' in sname:
                    if z.shape[1] > 100:
                        train_K = np.load(ROOT_PATH+'/mnist_precomp/{}_train_K0.npy'.format(sname))
                        train_K = (torch.exp(-train_K/a**2/2)+torch.exp(-train_K/a**2*50)+torch.exp(-train_K/a**2/200))/3
                        dev_K = np.load(ROOT_PATH+'/mnist_precomp/{}_dev_K0.npy'.format(sname))
                        dev_K = (torch.exp(-dev_K/a**2/2)+torch.exp(-dev_K/a**2*50)+torch.exp(-dev_K/a**2/200))/3
                    else:
                        train_K = (kernel(z, None, a, 1)+kernel(z, None, a/10, 1)+kernel(z, None, a*10, 1))/3
                        dev_K = (kernel(dev_z, None, a, 1)+kernel(dev_z, None, a/10, 1)+kernel(dev_z, None, a*10, 1))/3

                dev_err = my_loss(net(dev_x), dev_y, None, dev_K)
                err_in_expectation, mse = conditional_expected_loss(net=net, ax=ax, w_samples=w_samples, y_samples=y_samples, y_axz=y_axz, x_on=False)
                print('test', test_err, 'dev', dev_err, 'err_in_expectation', err_in_expectation, 'mse: ', mse)
                test_errs.append(test_err)
                dev_errs.append(dev_err)
                exp_errs.append(err_in_expectation)
                mse_s.append(mse)

                if es.step(dev_err):
                    break
            losses = {'test': test_errs, 'dev': dev_errs, 'exp': exp_errs, 'mse_': mse_s}
        return es.best, epoch, net, losses

    def get_causal_effect(net, do_A, w):
        """
        :param net: FCNN object
        :param do_A: a numpy array of interventions, size = B_a
        :param w: a torch tensor of w samples, size = B_w
        :return: a numpy array of interventional parameters
        """
        net.eval()
        # raise ValueError('have not tested get_causal_effect.')
        EYhat_do_A = []
        for a in do_A:
            a = np.repeat(a, [w.shape[0]]).reshape(-1,1)
            a_tensor = torch.as_tensor(a).float()
            w = w.reshape(-1,1).float()
            aw = torch.cat([a_tensor,w], dim=-1)
            aw_tensor = torch.tensor(aw)
            mean_h = torch.mean(net(aw_tensor)).reshape(-1, 1)
            EYhat_do_A.append(mean_h)
            print('a = {}, beta_a = {}'.format(np.mean(a), mean_h))
        return torch.cat(EYhat_do_A).detach().cpu().numpy()


    if training is True:
        print('training')
        for rep in range(3):
            print('*******REP: {}'.format(rep))
            save_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep, lr_id, dw_id, AW_train.shape[0]))
            # if os.path.exists(save_path):
            #    continue
            lr, dw = lrs[lr_id], decay_weights[dw_id]
            print('lr, dw', lr, dw)
            t0 = time.time()
            err, _, net, losses = fit(X[:n_train], Y[:n_train], Z[:n_train], X[n_train:], Y[n_train:], Z[n_train:], a, lr, dw,
                              ax=ax, y_axz=y_axz, w_samples=w_samples)
            t1 = time.time()-t0
            np.save(folder+'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep, lr_id, dw_id, AW_train.shape[0]), t1)
            g_pred = net(test_X).detach().numpy()
            test_err = ((g_pred-test_Y.numpy())**2).mean()
            np.savez(save_path, err=err.detach().numpy(), lr=lr, dw=dw, g_pred=g_pred, test_err=test_err)

            # make loss curves
            for (name, ylabel) in [('test', 'test av ||y - h||^2'), ('dev', 'R_V'), ('exp', 'E[y-h|a,z,x]'), ('mse_', 'mse_alternative_sim')]:
                errs = losses[name]
                stps = [50 + i * 5 for i in range(len(errs))]
                plt.figure()
                plt.plot(stps, errs)
                plt.xlabel('epoch')
                plt.ylabel(ylabel)
                plt.savefig(os.path.join(folder, name + '_{}_{}_{}_{}'.format(rep, lr_id, dw_id, AW_train.shape[0]) + '.png'))
                plt.close()

            # do causal effect estimates
            do_A = np.load(ROOT_PATH+"/data/zoo/"+sname+'/do_A.npz')['do_A']
            EY_do_A_gt = np.load(ROOT_PATH+"/data/zoo/"+sname+'/do_A.npz')['gt_EY_do_A']
            w_sample = train.w
            EYhat_do_A = get_causal_effect(net, do_A=do_A, w=w_sample)
            plt.figure()
            plt.plot([i+1 for i in range(20)], EYhat_do_A)
            plt.xlabel('A')
            plt.ylabel('EYdoA-est')
            plt.savefig(
                os.path.join(folder, 'causal_effect_estimates_{}_{}_{}'.format(lr_id, dw_id, AW_train.shape[0]) + '.png'))
            plt.close()

            print('ground truth ate: ', EY_do_A_gt)
            visualise_ATEs(EY_do_A_gt, EYhat_do_A,
                           x_name='E[Y|do(A)] - gt',
                           y_name='beta_A',
                           save_loc=folder,
                           save_name='ate_{}_{}_{}_{}.png'.format(rep, lr_id, dw_id, AW_train.shape[0]))
            causal_effect_mean_abs_err = np.mean(np.abs(EY_do_A_gt - EYhat_do_A))
            causal_effect_mae_file = open(os.path.join(folder, "ate_mae_{}_{}_{}.txt".format(lr_id, dw_id, AW_train.shape[0])), "a")
            causal_effect_mae_file.write("mae_rep_{}: {}\n".format(rep, causal_effect_mean_abs_err))
            causal_effect_mae_file.close()

    else:
        print('test')
        opt_res = []
        times = []
        for rep in range(10):
            res_list = []
            other_list = []
            times2 = []
            for lr_id in range(len(lrs)):
                for dw_id in range(len(decay_weights)):
                    load_path = os.path.join(folder, 'mmr_iv_nn_{}_{}_{}_{}.npz'.format(rep,lr_id,dw_id,datasize))
                    if os.path.exists(load_path):
                        res = np.load(load_path)
                        res_list += [res['err'].astype(float)]
                        other_list += [[res['lr'].astype(float),res['dw'].astype(float),res['test_err'].astype(float)]]
                    time_path = folder+'mmr_iv_nn_{}_{}_{}_{}_time.npy'.format(rep,lr_id,dw_id,datasize)
                    if os.path.exists(time_path):
                        t = np.load(time_path)
                        times2 += [t]
            res_list = np.array(res_list)
            other_list = np.array(other_list)
            other_list = other_list[res_list>0]
            res_list = res_list[res_list>0]
            optim_id = np.argsort(res_list)[0]# np.argmin(res_list)
            print(rep,'--',other_list[optim_id],np.min(res_list))
            opt_res += [other_list[optim_id][-1]]
        print('time: ', np.mean(times),np.std(times))
        print(np.mean(opt_res),np.std(opt_res))



if __name__ == '__main__': 
    # scenarios = ["step", "sin", "abs", "linear"]
    scenarios = ["sim_1d_no_x"]
    # index = int(sys.argv[1])
    # datasize = int(sys.argv[2])
    # sid,index = divmod(index,21)
    # lr_id, dw_id = divmod(index,7)
    for datasize in [5000]:  # [200, 2000]:
        for s in scenarios:
            for lr_id in range(3):
                for dw_id in range(7):
                    run_experiment_nn(s, datasize, [lr_id, dw_id])

        for s in scenarios:
            run_experiment_nn(s, datasize, [1, 0], training=False)
