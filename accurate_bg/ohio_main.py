import argparse
import json
import os

import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn_ohio import regressor, regressor_transfer, test_ckpt
from data_reader import DataReader


def personalized_train_ohio(epoch, ph, l_type="mae", path="../output"):
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
    # add test data of 2018 patient
    use_2018_test = True
    test_data_2018 = []
    for pid in pid_2018:
        reader = DataReader(
            "ohio", f"../data/OhioT1DM/2018/test/{pid}-ws-testing.xml", 5
        )
        test_data_2018 += reader.read()

    # a dumb dataset instance
    train_dataset = CGMSDataSeg(
        "ohio", "../data/OhioT1DM/2018/train/559-ws-training.xml", 5
    )
    sampling_horizon = 7
    prediction_horizon = ph
    scale = 0.01
    outtype = "Same"
    # train on training dataset
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    with open(os.path.join(path, "config.json")) as json_file:
        config = json.load(json_file)
    argv = (
        config["k_size"],
        config["nblock"],
        config["nn_size"],
        config["nn_layer"],
        config["learning_rate"],
        config["batch_size"],
        epoch,
        config["beta"],
    )
    # test on patients data
    outdir = os.path.join(path, f"ph_{prediction_horizon}_{l_type}")
    if not os.path.exists(outdir):
        os.makedirs(outdir)
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
            if use_2018_test:
                local_train_data += test_data_2018
            for k in train_pids:
                local_train_data += train_data[k]
            print(f"Pretrain data: {sum([sum(x) for x in local_train_data])}")
            train_dataset.data = local_train_data
            train_dataset.set_cutpoint = -1
            train_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                100,
                False,
                outtype,
                1,
                True,
            )
            regressor(train_dataset, *argv, l_type)
            # fine-tune on personal data
            target_test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            target_test_dataset.set_cutpoint = 1
            target_test_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                0.01,
                False,
                outtype,
                1,
                True,
            )
            target_train_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/train/{pid}-ws-training.xml", 5
            )

            target_train_dataset.set_cutpoint = -1
            target_train_dataset.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                100,
                False,
                outtype,
                1,
                True,
            )
            err, labels = test_ckpt(target_test_dataset)
            errs = [err]
            transfer_res = [labels]
            for i in range(1, 4):
                err, labels = regressor_transfer(
                    target_train_dataset,
                    target_test_dataset,
                    config["batch_size"],
                    epoch,
                    i,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--prediction_horizon", type=int, default=6)
    parser.add_argument("--outdir", type=str, default="../ohio_results")
    args = parser.parse_args()

    personalized_train_ohio(args.epoch, args.prediction_horizon, "mae", args.outdir)


if __name__ == "__main__":
    main()
