import argparse
import os

import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader import DataReader


def personalized_train_ohio(epoch, ph, l_type="mae"):
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
    prediction_horizon = ph
    scale = 0.01
    outtype = "Same"
    batch_size = 64
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    argv = (4, 4, 10, 2, 1e-3, batch_size, epoch, 1e-4)
    # test on patients data
    outdir = f"../ohio_results/ph_{prediction_horizon}_{l_type}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    all_errs = []
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            # only check results of 2020 patients
            if pid not in pid_2020:
                continue
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
            regressor(train_dataset, *argv, l_type)
            # fine-tune on personal data
            target_test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            target_test_dataset.set_cutpoint = 1
            target_test_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 0.01, False, outtype, 1
            )
            target_train_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )

            target_train_dataset.set_cutpoint = -1
            target_train_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
            )
            err, labels = test_ckpt(target_test_dataset)
            errs = [err]
            transfer_res = [labels]
            for i in range(1, 4):
                err, labels = regressor_transfer(
                    target_train_dataset, target_test_dataset, batch_size, epoch, i
                )
                errs.append(err)
                transfer_res.append(labels)
            transfer_res = np.concatenate(transfer_res, axis=1)
            np.savetxt(
                f"{outdir}/{pid}.txt",
                transfer_res,
                fmt="%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
            )
            all_errs.append([pid] + errs)
    all_errs = np.array(all_errs)
    np.savetxt(f"{outdir}/errors.txt", all_errs, fmt="%d %.4f %.4f %.4f %.4f")


def train_ohio(train, epoch):
    # read in all patients data
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
    # test on patients data
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
    parser.add_argument("--epoch", type=int, default=2)
    args = parser.parse_args()

    for l_type in ["mse", "mape", "mae", "relative_mse"]:
        for ph in [6, 12]:
            personalized_train_ohio(args.epoch, ph, l_type)


if __name__ == "__main__":
    main()
