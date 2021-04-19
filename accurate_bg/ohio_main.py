import argparse
import os

import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader import DataReader


def personalized_train_ohio(epoch):
    # read in all patients data
    pid_2018 = [559, 563, 570, 588, 575, 591]
    pid_2020 = [540, 552, 544, 567, 584, 596]
    pid_year = {2018: pid_2018, 2020: pid_2020}
    train_data = dict()
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            reader = DataReader(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )
            train_data[pid] = reader.read()
    print(f"Total training time sequences: {len(train_data)}")
    # a dumb dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "../data/OhioT1DM/2018/train/559-ws-training.xml", 5
    )
    sampling_horizon = 7
    prediction_horizon = 12
    scale = 0.01
    outtype = "Same"
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    argv = (4, 4, 10, 2, 1e-3, 64, epoch, 1e-4)
    # test on patients data
    outdir = f"../ohio_results/ph_{prediction_horizon}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    errs = []
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            # 100 is dumb if set_cutpoint is used
            train_pids = set(pid_2018 + pid_2020) - set([pid])
            local_train_data = []
            for k in train_pids:
                local_train_data += train_data[k]
            train_dataset.data = local_train_data
            train_dataset.set_cutpoint = -1
            train_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
            )
            regressor(train_dataset, *argv)
            # fine-tune on personal data TODO
            test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            test_dataset.set_cutpoint = 1
            test_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 0.01, False, outtype, 1
            )
            err, labels = test_ckpt(test_dataset)
            np.savetxt(f"{outdir}/{pid}.txt", labels, fmt="%.4f %.4f")
            errs.append([pid, err])
    errs = np.array(errs)
    np.savetxt(f"{outdir}/errors.txt", errs, fmt="%d %.4f")


def train_ohio(train, epoch):
    # read in all patients data TODO
    pid_2018 = [559, 563, 570, 588, 575, 591]
    pid_2020 = [540, 552, 544, 567, 584, 596]
    pid_year = {2018: pid_2018, 2020: pid_2020}
    train_data = []
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            reader = DataReader(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )
            train_data += reader.read()
    print(f"Total training time sequences: {len(train_data)}")
    # a pseudo dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "../data/OhioT1DM/2018/train/559-ws-training.xml", 5
    )
    train_dataset.data = train_data
    train_dataset.set_cutpoint = -1
    sampling_horizon = 7
    prediction_horizon = 12
    scale = 0.01
    outtype = "Same"
    # 100 is dumb if set_cutpoint is used
    train_dataset.reset(
        sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
    )
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    if train:
        argv = (4, 4, 10, 2, 1e-3, 64, epoch, 1e-4)
        regressor(train_dataset, *argv)
    # test on patients data TODO
    outdir = f"../ohio_results/ph_{prediction_horizon}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    errs = []
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            test_dataset.set_cutpoint = 1
            test_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 0.01, False, outtype, 1
            )
            err, labels = test_ckpt(test_dataset)
            np.savetxt(f"{outdir}/{pid}.txt", labels, fmt="%.4f %.4f")
            errs.append([pid, err])
    errs = np.array(errs)
    np.savetxt(f"{outdir}/errors.txt", errs, fmt="%d %.4f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    args = parser.parse_args()

    # train_ohio(args.train, args.epoch)
    personalized_train_ohio(args.epoch)


if __name__ == "__main__":
    main()
