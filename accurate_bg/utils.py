import os

import numpy as np
import pandas as pd


def get_result(ph=6, ind=1):
    path = f"../ohio_results/ph_{ph}"
    # pids = [552, 544, 567, 584, 596, 559,
    #       563, 570, 588, 575, 591, 540]
    pids = [596, 584, 567, 552, 544, 540]
    maes = []
    rmses = []
    for pid in pids:
        arr = np.loadtxt(os.path.join(path, str(pid) + ".txt"))
        mae = np.mean(np.abs(arr[:, 1:8:2] - arr[:, 0:8:2]), axis=0)
        rmse = np.sqrt(np.mean(np.power(arr[:, 1:8:2] - arr[:, 0:8:2], 2), axis=0))
        maes.append(mae)
        rmses.append(rmse)
    maes = np.array(maes)
    rmses = np.array(rmses)
    print(maes, rmses)
    return 100 * np.mean(maes[:, ind]), 100 * np.mean(rmses[:, ind])


def get_pbp_result(ph=6, ind=2):
    path = f"../ohio_results/ph_{ph}"
    # pids = [552, 544, 567, 584, 596, 559,
    #       563, 570, 588, 575, 591, 540]
    pids = [596, 584, 567, 552, 544, 540]
    data = []
    for pid in pids:
        arr = np.loadtxt(os.path.join(path, str(pid) + ".txt"))
        data.append(arr)
    data = np.concatenate(data, axis=0)
    mae = np.mean(np.abs(data[:, 1:8:2] - data[:, 0:8:2]), axis=0)
    rmse = np.sqrt(np.mean(np.power(data[:, 1:8:2] - data[:, 0:8:2], 2), axis=0))
    return 100 * mae[ind], 100 * rmse[ind]


def compare_result():
    path = "../ohio_results/challenge.csv"
    df = pd.read_csv(path)
    mae1, rmse1 = get_pbp_result()
    mae2, rmse2 = get_pbp_result(12)
    df = df.append(
        {
            "Paper ID": "ours",
            "30min_MAE": mae1,
            "60min_MAE": mae2,
            "30min_RMSE": rmse1,
            "60min_RMSE": rmse2,
        },
        ignore_index=True,
    )
    df["overall"] = (
        df["30min_RMSE"] + df["30min_MAE"] + df["60min_RMSE"] + df["60min_MAE"]
    )
    df["30 min"] = df["30min_RMSE"] + df["30min_MAE"]
    df["60 min"] = df["60min_RMSE"] + df["60min_MAE"]
    df["MAE"] = df["60min_MAE"] + df["30min_MAE"]
    df["RMSE"] = df["60min_RMSE"] + df["30min_RMSE"]
    print(df)
    for col in list(df.columns):
        if col == "Paper ID":
            continue
        new_df = df.sort_values(col, ignore_index=True)
        # print(new_df)
        print(col, new_df.index[new_df["Paper ID"] == "ours"])


def main():
    # get_pbp_result()
    compare_result()


if __name__ == "__main__":
    main()
