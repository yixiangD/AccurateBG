import numpy as np
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
    # print(train_data)
    # print(len(train_data))
    # train on patients data TODO


if __name__ == "__main__":
    main()
