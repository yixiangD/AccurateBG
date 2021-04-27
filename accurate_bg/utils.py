import os

import numpy as np
import pandas as pd


def get_result(l_type="mae", ph=6, ind=0, path="ohio_results", standard=True):
    path = f"{path}/ph_{ph}_{l_type}"
    # pids = [552, 544, 567, 584, 596, 559,
    #       563, 570, 588, 575, 591, 540]
    pids = [
        540,
        544,
        552,
        567,
        584,
        596,
    ]
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
    if standard:
        coeff = 60.65
    else:
        coeff = 100
    best_maes = coeff * maes[:, ind]
    best_rmses = coeff * rmses[:, ind]
    df = pd.DataFrame(
        {"PID": pids, f"{ph*5}min MAE": best_maes, f"{ph*5}min RMSE": best_rmses}
    )
    return df


def get_pbp_result(ph=6, l_type="mse", ind=2):
    path = f"../ohio_results/ph_{ph}_{l_type}"
    # pids = [552, 544, 567, 584, 596, 559,
    #       563, 570, 588, 575, 591, 540]
    pids = [
        540,
        544,
        552,
        567,
        584,
        596,
    ]
    data = []
    for pid in pids:
        arr = np.loadtxt(os.path.join(path, str(pid) + ".txt"))
        data.append(arr)
    data = np.concatenate(data, axis=0)
    mae = np.mean(np.abs(data[:, 1:8:2] - data[:, 0:8:2]), axis=0)
    rmse = np.sqrt(np.mean(np.power(data[:, 1:8:2] - data[:, 0:8:2], 2), axis=0))
    return 100 * mae[ind], 100 * rmse[ind]


def compare_result(l_type):
    path = "../ohio_results/challenge.csv"
    df = pd.read_csv(path)
    mae1, rmse1 = get_pbp_result(6, l_type)
    mae2, rmse2 = get_pbp_result(12, l_type)
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
    # print(df)
    for col in list(df.columns):
        if col == "Paper ID":
            continue
        new_df = df.sort_values(col, ignore_index=True)
        if col == "MAE":
            print(new_df)
        print(col, new_df.index[new_df["Paper ID"] == "ours"])


def compare_only_bg_result(
    l_type="mae", transfer=2, path="../ohio_results", standard=True
):
    res_30 = get_result(l_type, 6, transfer, path, standard)
    res_60 = get_result(l_type, 12, transfer, path, standard)
    res = pd.merge(res_30, res_60, how="left", on="PID")
    path = "../ohio_results/bg_ohio.xlsx"
    peers = ["khadem", "bevan", "joedicke", "ma"]
    result = dict()
    result["metric"] = ["30min MAE", "30min RMSE", "60min MAE", "60min RMSE"]
    result["ours"] = res.mean().to_numpy()[1:]
    for p in peers:
        df = pd.read_excel(path, sheet_name=p)
        result[p + " et al."] = df.mean().to_numpy()[1:]
    result = pd.DataFrame(result)
    result.to_csv("../ohio_results/comparison.csv", float_format="%.4f", index=False)
    print(result)
    print(result.sum())


def main():
    # get_pbp_result()
    # compare_only_bg_result("mae", 2, f"../ohio_results")
    compare_only_bg_result("mae", 2, "../output", True)
    exit()
    for l_type in ["mse", "mape", "mae", "relative_mse"]:
        print(l_type)
        compare_result(l_type)


if __name__ == "__main__":
    main()
