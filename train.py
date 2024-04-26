import math
import os
import random
import warnings
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset

from work_ADNI_V1.model1 import model_all

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def stest(model, datasets_test, num_win):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    pro_all = []
    model.eval()
    with torch.no_grad():
        for fmri, net, tg, label in datasets_test:
            fmri, net, tg, label = fmri.to(DEVICE), net.to(DEVICE), tg.to(DEVICE), label.to(DEVICE)
            fmri = fmri.float()
            net = net.float()
            tg = tg.float()
            label = label.long()
            # print(label.shape)
            outs = model(fmri, net, tg)
            losss = F.nll_loss(outs, label)
            eval_loss += float(losss)
            gailv, pred = outs.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / net.shape[0]
            eval_acc += acc
            pre = pred.cpu().detach().numpy()
            pre_all.extend(pre)
            label_true = label.cpu().detach().numpy()
            labels_all.extend(label_true)
            pro_all.extend(outs[:, 1].cpu().detach().numpy())

        tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        eval_acc_epoch = accuracy_score(labels_all, pre_all)
        precision = precision_score(labels_all, pre_all)
        recall = recall_score(labels_all, pre_all)
        f1 = f1_score(labels_all, pre_all)
        my_auc = roc_auc_score(labels_all, pro_all)

    return eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all


def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x


wd = 1e-3
log = open('SMC_EMCI_ADNI.txt', mode='a', encoding='utf-8')

for he_dim in [4, 8, 16, 32, 64, 128]:
    for yu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for drop in [0.6]:
            for lr1 in [5e-4, 1e-3, 3e-3, 5e-3, 7e-3, 9e-3]:

                ##################################  NC  vs   ill########################################
                #
                m = loadmat('D:\codespace\work5\datasets\Multimodal_ADNI_fMRI&ADNI.mat')  # fmri
                keysm = list(m.keys())
                fdata = m[keysm[3]]
                labels = m[keysm[5]][0]
                for i in range(203):
                    max_t = np.max(fdata[i])
                    min_t = np.min(fdata[i])
                    fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
                for i in range(labels.shape[0]):
                    if labels[i] == 2:
                        labels[i] = 1

                ################################# NC  vs   SMC########################################
                # m = loadmat('D:\codespace\work5\datasets\Multimodal_ADNI_fMRI&ADNI.mat')  # fmri
                # keysm = list(m.keys())
                # fdata = m[keysm[3]][0:143]

                # labels = m[keysm[5]][0][0:143]
                # for i in range(fdata.shape[0]):
                #     max_t = np.max(fdata[i])
                #     min_t = np.min(fdata[i])
                #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
                #################################  NC  vs   EMCI########################################

                # m = loadmat('D:\codespace\work5\datasets\Multimodal_ADNI_fMRI&ADNI.mat')  # fmri
                # keysm = list(m.keys())
                # fdata = m[keysm[3]]
                # for i in range(fdata.shape[0]):
                #     max_t = np.max(fdata[i])
                #     min_t = np.min(fdata[i])
                #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
                # fdata = torch.cat((torch.tensor(fdata[0:73]), torch.tensor(fdata[143:203])))
                # fdata = fdata.numpy()
                # labels = m[keysm[5]][0]
                # labels = torch.cat((torch.tensor(labels[0:73]), torch.tensor(labels[143:203])))
                # labels = labels.numpy()
                #
                # for i in range(labels.shape[0]):
                #     if labels[i] == 2:
                #         labels[i] = 1
                #################################   SMC  vs   EMCI########################################
                # m = loadmat('D:\codespace\work5\datasets\Multimodal_ADNI_fMRI&ADNI.mat')  # fmri
                # keysm = list(m.keys())
                # fdata = m[keysm[3]]
                # fdata = fdata[73:203]
                # for i in range(fdata.shape[0]):
                #     max_t = np.max(fdata[i])
                #     min_t = np.min(fdata[i])
                #     fdata[i] = MaxMinNormalization(fdata[i], max_t, min_t)
                #
                # labels = m[keysm[5]][0]
                # labels = labels[73:203]
                # for i in range(labels.shape[0]):
                #     if labels[i] == 2:
                #         labels[i] = 0


                index = [i for i in range(fdata.shape[0])]
                np.random.shuffle(index)
                fdata = fdata[index]
                labels = labels[index]


                #######################################DFBN##########################################
                def construct_DFCN(dataset, num_window, yu):
                    fmri_all = []
                    nets_all = []
                    win_length = dataset.shape[2] // num_window
                    for i in range(dataset.shape[0]):
                        fmri = []
                        nets = []
                        datas = dataset[i]  # 90*240
                        for j in range(num_window):
                            window_fmri = datas[:, win_length * j:win_length * (j + 1)]
                            fmri.append(window_fmri)
                            net = np.abs(np.corrcoef(window_fmri))
                            net[net < yu] = 0
                            max_t = np.max(net)
                            min_t = np.min(net)
                            net = MaxMinNormalization(net, max_t, min_t)
                            nets.append(net)
                        fmri_all.append(fmri)
                        nets_all.append(nets)
                    return fmri_all, nets_all  # torch.Size([306, 6, 90, 40])


                fmri_all, nets_all = construct_DFCN(fdata,5 , yu)
                fmri_all = np.array(fmri_all)
                nets_all = np.array(nets_all)


                ####################################### Temporal Graph#######################################


                def construct_TG(dyfnet, win, yu):
                    all_TG = []
                    for i in range(dyfnet.shape[0]):
                        dfc = dyfnet[i]
                        net_TG = np.zeros((win * 90, win * 90))
                        for t in range(win):
                            start_row = t * 90
                            start_col = t * 90

                            net_TG[start_row:start_row + 90, start_col:start_col + 90] = dfc[t]
                            if t + 1 < win:
                                for tt in range(t + 1, win):
                                    w = math.exp(-(abs((t - tt) / win)))
                                    start_row = t * 90
                                    start_col = tt * 90
                                    net_TG[start_row:start_row + 90, start_col:start_col + 90] = dfc[t] * w
                                    net_TG[start_col:start_col + 90, start_row:start_row + 90] = dfc[t] * w
                        net_TG[net_TG < yu] = 0
                        max_t = np.max(net_TG)
                        min_t = np.min(net_TG)
                        net_TG = MaxMinNormalization(net_TG, max_t, min_t)

                        all_TG.append(net_TG)
                    return all_TG


                all_TG = construct_TG(nets_all, 5, yu)
                all_TG = np.array(all_TG)


                # print(all_TG.shape)

                class ADNI(Dataset):
                    def __init__(self):
                        super(ADNI, self).__init__()
                        self.feas = fmri_all
                        self.nets = nets_all
                        self.tgs = all_TG
                        self.label = labels

                    def __getitem__(self, item):
                        fea = self.feas[item]
                        net = self.nets[item]
                        tg = self.tgs[item]
                        label = self.label[item]
                        return fea, net, tg, label

                    def __len__(self):
                        return self.feas.shape[0]


                num_win = 2
                n_class = 2
                avg_acc = 0
                avg_spe = 0
                avg_recall = 0
                avg_f1 = 0
                avg_auc = 0
                avg_sens = 0
                avg_spec = 0
                pre_ten = []
                label_ten = []
                pro_ten = []
                test_acc = []
                test_pre = []
                test_recall = []
                test_f1 = []
                test_auc = []
                test_sens = []
                test_spec = []
                dataset = ADNI()
                k = 10
                i = 0

                KF = KFold(n_splits=k, shuffle=True, random_state=7)
                for train_idx, test_idx in KF.split(dataset):
                    train_subsampler = SubsetRandomSampler(train_idx)
                    test_sunsampler = SubsetRandomSampler(test_idx)
                    datasets_train = DataLoader(dataset, batch_size=20, shuffle=False, sampler=train_subsampler,
                                                drop_last=True)
                    datasets_test = DataLoader(dataset, batch_size=20, shuffle=False, sampler=test_sunsampler,
                                               drop_last=True)
                    min_loss = 1e10
                    losses = []
                    acces = []
                    eval_losses = []
                    eval_acces = []
                    patience = 0
                    patiences = 30
                    min_acc = 0
                    pre_gd = 0
                    recall_gd = 0
                    f1_gd = 0
                    auc_gd = 0
                    sens_gd = 0
                    spec_gd = 0
                    labels_all_gd = 0
                    pro_all_gd = 0
                    model = model_all(39, 20, 5, he_dim)
                    model.to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=1e-3)
                    for e in range(2):
                        model.train()
                        train_loss = 0
                        train_acc = 0
                        pre_all_train = []
                        labels_all_train = []
                        model.train()
                        for fmri, net, tg, label in datasets_train:
                            fmri, net, tg, label = fmri.to(DEVICE), net.to(DEVICE), tg.to(DEVICE), label.to(
                                DEVICE)
                            fmri = fmri.float()
                            net = net.float()
                            tg = tg.float()
                            label = label.long()
                            out = model(fmri, net, tg)
                            loss_c = F.nll_loss(out, label)

                            optimizer.zero_grad()
                            loss_c.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                            optimizer.step()
                            train_loss += float(loss_c)
                            _, pred = out.max(1)
                            pre = pred.cpu().detach().numpy()
                            pre_all_train.extend(pre)
                            label_true = label.cpu().detach().numpy()
                            labels_all_train.extend(label_true)
                        losses.append(train_loss / len(datasets_train))
                        acces.append(train_acc / len(datasets_train))
                        train_acc = accuracy_score(labels_all_train, pre_all_train)
                        eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, my_auc, sensitivity, specificity, pre_all, labels_all, pro_all = stest(
                            model, datasets_test, num_win)

                        if eval_acc_epoch > min_acc:
                            torch.save(model.state_dict(), './latest' + str(i) + '.pth')
                            print("Model saved at epoch{}".format(e))
                            min_acc = eval_acc_epoch
                            pre_gd = precision
                            recall_gd = recall
                            f1_gd = f1
                            auc_gd = my_auc
                            sens_gd = sensitivity
                            spec_gd = specificity
                            labels_all_gd = labels_all
                            pro_all_gd = pro_all
                            patience = 0


                        eval_losses.append(eval_loss / len(datasets_test))
                        eval_acces.append(eval_acc / len(datasets_test))
                        print(
                            'i:{},epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f},precision : {'
                            ':.6f},recall : {:.6f},f1 : {:.6f},auc : {:.6f} '
                            .format(i, e, train_loss / len(datasets_train), train_acc,
                                    eval_loss / len(datasets_test),
                                    eval_acc_epoch,
                                    precision, recall, f1, my_auc))
                    test_acc.append(min_acc)
                    test_pre.append(pre_gd)
                    test_recall.append(recall_gd)
                    test_f1.append(f1_gd)
                    test_auc.append(auc_gd)
                    test_sens.append(sens_gd)
                    test_spec.append(spec_gd)
                    label_ten.extend(labels_all_gd)
                    pro_ten.extend(pro_all_gd)

                    i = i + 1
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_acc",
                      test_acc, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_pre",
                      test_pre, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_recall",
                      test_recall, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_f1",
                      test_f1, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_auc",
                      test_auc, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_sens",
                      test_sens, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "test_spec",
                      test_spec, file=log)
                avg_acc = sum(test_acc) / k
                avg_pre = sum(test_pre) / k
                avg_recall = sum(test_recall) / k
                avg_f1 = sum(test_f1) / k
                avg_auc = sum(test_auc) / k
                avg_sens = sum(test_sens) / k
                avg_spec = sum(test_spec) / k
                print("*****************************************************", file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      'acc', avg_acc,
                      file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      'pre', avg_pre,
                      file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      'recall',
                      avg_recall, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      'f1', avg_f1,
                      file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      'auc', avg_auc,
                      file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "sensitivity",
                      avg_sens, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "specificity",
                      avg_spec, file=log)

                acc_std = np.sqrt(np.var(test_acc))
                pre_std = np.sqrt(np.var(test_pre))
                recall_std = np.sqrt(np.var(test_recall))
                f1_std = np.sqrt(np.var(test_f1))
                auc_std = np.sqrt(np.var(test_auc))
                sens_std = np.sqrt(np.var(test_sens))
                spec_std = np.sqrt(np.var(test_spec))
                print("*****************************************************", file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "acc_std",
                      acc_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "pre_std",
                      pre_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "recall_std",
                      recall_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "f1_std",
                      f1_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "auc_std",
                      auc_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "sens_std",
                      sens_std, file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      "spec_std",
                      spec_std, file=log)
                print("*****************************************************", file=log)

                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      label_ten,
                      file=log)
                print("num_win", num_win, 'he_dim', he_dim, 'yu', yu, 'drop', drop, 'lr', lr1, 'wd', wd,
                      pro_ten,
                      file=log)
                print("*****************************************************", file=log)
