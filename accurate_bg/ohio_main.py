import numpy as np
from cgms_data_seg import CGMSDataSeg
from cnn import regressor, regressor_transfer
from data_reader import DataReader


def main():
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
    prediction_horizon = 6
    scale = 0.01
    outtype = "Same"
    # 100 is dumb if set_cutpoint is used
    train_dataset.reset(
        sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
    )
    # test on patients data TODO
    # k_size, nblock, nn_size, nn_layer, learning_rate, batch_size, epoch, beta
    argv = (4, 4, 10, 2, 1e-3, 64, 50, 1e-4)
    for year in list(pid_year.keys()):
        pids = pid_year[year]
        for pid in pids:
            test_dataset = CGMSDataSeg(
                "ohio", f"../data/OhioT1DM/{year}/test/{pid}-ws-testing.xml", 5
            )
            test_dataset.reset(
                sampling_horizon, prediction_horizon, scale, 0.01, False, outtype, 1
            )
            erro, labs = regressor(train_dataset, test_dataset, *argv)


if __name__ == "__main__":
    main()
