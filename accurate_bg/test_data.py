from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from data_reader import DataReader
from helper import read_patient_info


def test_ohio():
    pid = 559
    reader = DataReader("ohio", f"../data/OhioT1DM/2018/train/{pid}-ws-training.xml", 5)
    data = reader.read()
    plt.figure()
    tmax = 0
    for y in data:
        tot = len(y)
        plt.plot(np.arange(tot), y)
        tmax = max(tmax, tot)
        hyper = sum(np.array(y) > 180)
        hypo = sum(np.array(y) < 70)
        print("tot", tot, "num hyper", hyper, "num hypo", hypo)
    plt.hlines(70, 0, tmax)
    plt.hlines(80, 0, tmax)
    plt.hlines(180, 0, tmax)
    plt.show()


def test_va():
    # reader = DataReader("direcnet_pid", "../data/tblADataCGMS.csv", 5)
    # data = reader.read()
    reader = DataReader("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)
    data = reader.read()
    normal, diabetic = read_patient_info()
    pids = list(normal["Patient ID"])
    t = np.arange(len(data[pids[0]][0])) * 5
    res = np.vstack((t, data[pids[0]][0]))
    np.savetxt("{}.txt".format(pids[0]), res, fmt="%.4f")

    for p in pids:
        m = map(lambda x: len(x), data[p])
        print(list(m))
    exit()
    plt.figure()
    tot, hyper, hypo = 0, 0, 0
    for pid in data:
        #    if pid not in pids:
        #        continue
        for y in data[pid]:
            t = np.arange(len(y)) * 5
            tot += len(y)
            y = np.array(y)
            hyper += sum(y > 180)
            hypo += sum(y < 70)
            plt.plot(t, y)
            plt.hlines(70, 0, 10000)
            plt.hlines(80, 0, 10000)
            plt.hlines(180, 0, 10000)
    plt.show()
    print(hypo, hyper, tot)


def main():
    test_ohio()


if __name__ == "__main__":
    main()
