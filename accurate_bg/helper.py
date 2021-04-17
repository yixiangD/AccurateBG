from multiprocessing import Pool

import numpy as np
import pandas as pd
from cgms_data_seg import CGMSDataSeg
from sklearn.model_selection import KFold


def hyperglycemia(x, threshold=1.8):
    return np.hstack((x >= threshold, x < threshold)).astype(np.float32)


def hypoglycemia(x, threshold=0.7):
    # threshold can be set to 0.54 suggested by Dr. Mantzoros
    return np.hstack((x <= threshold, x > threshold)).astype(np.float32)


def threeclasses(x, th_min=0.7, th_max=1.8):
    def safe(x):
        return [x[0] < th_max and x[0] > th_min]

    return np.hstack(
        (x <= th_min, np.apply_along_axis(safe, 1, x), x >= th_max)
    ).astype(np.float32)


def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def read_patient_info():
    df = pd.read_excel(
        open("../data/CGMdataCSMcomplete_update.xlsx", "rb"), sheet_name="Demographics"
    )
    # trim empty spaces in front and after cell value
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return df[df["Group"] == "Control"], df[df["Group"] != "Control"]


def read_direcnet_patient_info():
    df = pd.read_csv("../data/tblAScreening.csv")
    df2 = pd.read_csv("../data/tblALab.csv")
    df = df.merge(df2, on="PtID")
    subdf = df[["PtID", "Gender", "BMI", "Type1DM", "AgeAtBaseline", "HbA1c"]]
    return subdf[subdf["Type1DM"] != "Yes"], subdf[subdf["Type1DM"] == "Yes"]


def groupby_meds():
    control, diabetic = read_patient_info()
    cat1 = ["no dm meds", np.nan, "none"]
    cat2 = ["insulin", "lantus", "novolin", "hum"]
    no_med = diabetic[diabetic["Medications"].str.lower().isin(cat1)]
    insulin = pd.DataFrame()
    for item in cat2:
        insulin = pd.concat(
            [
                insulin,
                diabetic[
                    diabetic["Medications"].str.contains(item, na=False, case=False)
                ],
            ]
        )
    other_pid = (
        set(diabetic["Patient ID"])
        - set(no_med["Patient ID"])
        - set(insulin["Patient ID"])
    )
    other = diabetic[diabetic["Patient ID"].isin(other_pid)]
    no_med.drop_duplicates(inplace=True)
    insulin.drop_duplicates(inplace=True)
    other.drop_duplicates(inplace=True)
    return no_med, insulin, other


def runner(learner, argv, transfer=None, outtype="Same", cut_point=50, label="data"):
    # 'Same' for regressor; 'None' for classifier
    _, diabetic = read_patient_info()
    pids = list(diabetic["Patient ID"])
    low_fid_data = CGMSDataSeg("direcnet", "../data/tblADataCGMS.csv", 5)
    sampling_horizon = 7
    prediction_horizon = 6
    scale = 0.01
    train_test_ratio = 0.1  # dummy number

    low_fid_data.reset(
        sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
    )

    of = open("{}_{}.txt".format(label, cut_point), "a+")
    for pid in range(1, 6):
        print("=" * 100)
        if pid <= 3:
            print("pid: ", pid)
            high_fid_data = CGMSDataSeg(
                "VA", ["../data/CLARITY_Export_{}.txt".format(pid)], 5
            )
        elif pid == 4:
            print("pid: ", pid)
            high_fid_data = CGMSDataSeg("VA1", "../data/A_J_201211301041.xls", 5)
        else:
            high_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)
        if pid <= 4:
            high_fid_data.set_cutpoint = cut_point
            high_fid_data.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                train_test_ratio,
                False,
                outtype,
                1,
            )
            with Pool(1) as p:
                err = p.apply(learner, (low_fid_data, high_fid_data, *argv))
            if transfer is not None:
                with Pool(1) as p:
                    terr1 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 1))
                    terr2 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 2))
                    terr3 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 3))
                of.write(
                    "{:d} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                        pid, err, terr1, terr2, terr3
                    )
                )
            else:
                of.write("{:d} {:.4f}\n".format(pid, err))
        else:
            for item in high_fid_data.raw_data:
                if item in pids:
                    print("Processing pid: {}".format(item))
                    high_fid_data.data = high_fid_data.raw_data[item]
                    high_fid_data.set_cutpoint = cut_point
                    high_fid_data.reset(
                        sampling_horizon,
                        prediction_horizon,
                        scale,
                        train_test_ratio,
                        False,
                        outtype,
                        1,
                    )
                    if cut_point > high_fid_data.train_n:
                        print("Training data size smaller than required, skipped")
                        continue
                    with Pool(1) as p:
                        err = p.apply(learner, (low_fid_data, high_fid_data, *argv))
                    if transfer is not None:
                        with Pool(1) as p:
                            terr1 = p.apply(
                                transfer, (high_fid_data, argv[-3], argv[-2], 1)
                            )
                            terr2 = p.apply(
                                transfer, (high_fid_data, argv[-3], argv[-2], 2)
                            )
                            terr3 = p.apply(
                                transfer, (high_fid_data, argv[-3], argv[-2], 3)
                            )
                            of.write(
                                "{:s} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                                    item, err, terr1, terr2, terr3
                                )
                            )
                    else:
                        of.write("{:s} {:.4f}\n".format(item, err))
    of.close()


def hierarchical_runner(
    learner, argv, transfer=None, outtype="Same", cut_point=50, label="data", throw=5
):
    # 'Same' for regressor; 'None' for classifier
    _, diabetic = read_patient_info()
    pids = list(diabetic["Patient ID"])
    print(f"# {len(pids)} diabetic patients from new samples")
    # read direcnet
    low_fid_data = CGMSDataSeg("direcnet", "../data/tblADataCGMS.csv", 5)
    sampling_horizon = 7
    prediction_horizon = 6
    scale = 0.01
    train_test_ratio = 0.1  # dummy number

    # store patients' data in all_data
    all_data = dict()
    all_data["4"] = CGMSDataSeg("VA1", "../data/A_J_201211301041.xls", 5).data

    for pid in range(1, 4):
        all_data[str(pid)] = CGMSDataSeg(
            "VA", ["../data/CLARITY_Export_{}.txt".format(pid)], 5
        ).data

    # create an interface for high_fid_data
    high_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)
    all_data.update({pid: high_fid_data.raw_data[pid] for pid in pids})
    all_pids = list(all_data.keys())
    raw_fold = len(all_pids) / (len(all_pids) - throw)
    if raw_fold < 1.5:
        # for fold not meeting requirement
        fold = np.ceil(1 / (1 - 1 / raw_fold))
    else:
        fold = np.ceil(raw_fold)
    kf = KFold(n_splits=int(fold))
    print(f"# {len(all_pids)} patients, {raw_fold} fold requested, {fold} fold given")
    print(f"before adding cohort, size {len(low_fid_data.data)}")
    already_tested = []

    of = open(f"{label}_{throw}_{cut_point}.txt", "a+")

    for train_index, test_index in kf.split(all_pids):
        # map to get train pid
        if raw_fold < 1.5:
            train_index, test_index = test_index, train_index
        train_ids = [all_pids[k] for k in train_index]
        for k in train_ids:
            low_fid_data.data += all_data[k]
        print(f"after adding cohort, size {len(low_fid_data.data)}")
        low_fid_data.reset(
            sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
        )
        # map to get test pid set and test on new ids
        test_ids = set([all_pids[k] for k in test_index]) - set(already_tested)
        already_tested += list(test_ids)
        print(f"{throw} train ids: {train_ids}")
        print(f"{len(already_tested)} already tested: {already_tested}")
        print(f"{len(test_ids)} in testing: {test_ids}")
        for item in test_ids:
            print(f"Processing patient {item}")
            high_fid_data.data = all_data[item]
            high_fid_data.set_cutpoint = cut_point
            high_fid_data.reset(
                sampling_horizon,
                prediction_horizon,
                scale,
                train_test_ratio,
                False,
                outtype,
                1,
            )
            if cut_point > high_fid_data.train_n:
                print("Training data size smaller than required, skipped")
                continue
            with Pool(1) as p:
                err = p.apply(learner, (low_fid_data, high_fid_data, *argv))
                if transfer is not None:
                    with Pool(1) as p:
                        terr1 = p.apply(
                            transfer, (high_fid_data, argv[-3], argv[-2], 1)
                        )
                        terr2 = p.apply(
                            transfer, (high_fid_data, argv[-3], argv[-2], 2)
                        )
                        terr3 = p.apply(
                            transfer, (high_fid_data, argv[-3], argv[-2], 3)
                        )
                        of.write(
                            "{:s} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                                item, err, terr1, terr2, terr3
                            )
                        )
                else:
                    of.write("{:s} {:.4f}\n".format(item, err))
    of.close()


def native_runner(
    learner, argv, transfer=None, outtype="Same", cut_point=50, label="data"
):
    # 'Same' for regressor; 'None' for classifier
    # nomed, insulin, other = groupby_meds()
    _, allp = read_patient_info()
    the_group = allp
    # the_group = the_group.sample(frac=0.3, random_state=2)
    print(f"{len(the_group)} patients selected")
    # _, diabetic = read_patient_info()

    # pids = list(diabetic["Patient ID"])
    pids = list(the_group["Patient ID"])
    sampling_horizon = 7
    prediction_horizon = 6
    scale = 0.01
    train_test_ratio = 0.1  # dummy number

    # store patients' data in all_data
    all_data = dict()

    low_fid_data = CGMSDataSeg("VA1", "../data/A_J_201211301041.xls", 5)
    all_data["4"] = low_fid_data.data

    for pid in range(1, 4):
        data = CGMSDataSeg("VA", ["../data/CLARITY_Export_{}.txt".format(pid)], 5)
        all_data[str(pid)] = data.data

    # create an interface for high_fid_data
    high_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)
    all_data.update({pid: high_fid_data.raw_data[pid] for pid in pids})
    # all_pids = all_data.keys()  # not used

    of = open("{}_{}.txt".format(label, cut_point), "a+")
    for item in pids:
        # for item in all_pids:
        print("Processing pid: {}".format(item))
        train_pids = set(pids) - set(item)
        train_data = [all_data[k] for k in train_pids]
        low_fid_data.data = [item for data in train_data for item in data]
        print(
            "# {} data seg from other patients".format(
                sum([len(x) for x in low_fid_data.data])
            )
        )
        low_fid_data.set_cutpoint = -1
        low_fid_data.reset(
            sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
        )

        high_fid_data.data = all_data[item]
        high_fid_data.set_cutpoint = cut_point
        high_fid_data.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            train_test_ratio,
            False,
            outtype,
            1,
        )
        if cut_point > high_fid_data.train_n:
            print("Training data size smaller than required, skipped")
            continue
        with Pool(1) as p:
            err, labs = p.apply(learner, (low_fid_data, high_fid_data, *argv))
        if transfer is not None:
            with Pool(1) as p:
                terr1, score1 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 1)
                )
                terr2, score2 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 2)
                )
                terr3, score3 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 3)
                )
            of.write(
                "{:s} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                    item, err, terr1, terr2, terr3
                )
            )
            np.savetxt(
                f"{item}.txt", np.hstack((labs, score1, score2, score3)), fmt="%.4f"
            )
        else:
            of.write("{:s} {:.4f}\n".format(item, err))
    of.close()


def feature_runner(
    learner, argv, transfer=None, outtype="Same", cut_point=50, label="data"
):
    # 'Same' for regressor; 'None' for classifier
    nomed, insulin, other = groupby_meds()
    _, t2d = read_patient_info()
    the_group = t2d
    # the_group = the_group.sample(frac=0.3, random_state=2)
    print(f"{len(the_group)} patients selected")
    # the_group = pd.concat([no_med, other])
    pids = list(the_group["Patient ID"])
    bmis = pd.Series(the_group["BMI"].values, index=the_group["Patient ID"]).to_dict()

    genders = pd.Series(the_group["Gender"].values, index=the_group["Patient ID"])
    genders = genders.apply(lambda x: 100 if x == "Female" else 0).to_dict()

    # ages = pd.Series(the_group["Age"].values, index=the_group["Patient ID"]).to_dict()

    hba1cs = pd.Series(the_group["HbA1c"].values, index=the_group["Patient ID"])
    hba1cs.dropna(how="any", inplace=True)
    hba1cs = hba1cs.to_dict()

    sampling_horizon = 7
    prediction_horizon = 4
    scale = 0.01
    train_test_ratio = 0.01  # dummy number

    # create an interface for low_fid_data and clean out data
    low_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)

    # create an interface for high_fid_data
    high_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)
    # store diabetic patients' data in all_data
    all_data = {k: high_fid_data.raw_data[k] for k in pids}
    all_feature = {
        k: list(np.tile(bmis[k], (len(high_fid_data.raw_data[k]), 1)))
        for k in bmis.keys()
    }
    # all_feature = {k : list(np.tile(hba1cs[k], (len(high_fid_data.raw_data[k]), 1))) for k in hba1cs.keys()}

    of = open("{}_{}.txt".format(label, cut_point), "a+")
    for item in all_feature.keys():
        print("Processing pid: {}".format(item))
        low_fid_data.data = []
        low_fid_data.feature = []
        train_pids = set(all_feature.keys()) - set(item)
        for k in train_pids:
            low_fid_data.data += all_data[k]
            # set up feature vector
            low_fid_data.feature += all_feature[k]
        low_fid_data.set_cutpoint = -1
        low_fid_data.reset(
            sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
        )

        high_fid_data.data = all_data[item]
        high_fid_data.set_cutpoint = cut_point
        high_fid_data.feature = all_feature[item]
        high_fid_data.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            train_test_ratio,
            False,
            outtype,
            1,
        )
        if cut_point > high_fid_data.train_n:
            print("Training data size smaller than required, skipped")
            continue
        with Pool(1) as p:
            err, labs = p.apply(learner, (low_fid_data, high_fid_data, *argv))
        if transfer is not None:
            with Pool(1) as p:
                terr1, score1 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 1)
                )
                terr2, score2 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 2)
                )
                terr3, score3 = p.apply(
                    transfer, (high_fid_data, argv[-3], argv[-2], 3)
                )
            of.write(
                "{:s} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                    item, err, terr1, terr2, terr3
                )
            )
            np.savetxt(
                f"{item}.txt", np.hstack((labs, score1, score2, score3)), fmt="%.4f"
            )
        else:
            of.write("{:s} {:.4f}\n".format(item, err))
    of.close()


def hierarchical_feature_runner(
    learner, argv, transfer=None, outtype="Same", cut_point=50, label="data"
):
    # 'Same' for regressor; 'None' for classifier

    # create an interface for low_fid_data and clean out data
    low_fid_data = CGMSDataSeg("direcnet_pid", "../data/tblADataCGMS.csv", 5)

    # create an interface for high_fid_data
    high_fid_data = CGMSDataSeg("VA2", "../data/CGMdataCSMcomplete.xlsx", 5)

    sampling_horizon = 7
    prediction_horizon = 6
    scale = 0.01
    train_test_ratio = 0.1  # dummy number

    # Reading demographic data for BIDMC patients, t2d
    _, t2d = read_patient_info()
    bmis = pd.Series(t2d["BMI"].values, index=t2d["Patient ID"]).to_dict()
    genders = pd.Series(t2d["Gender"].values, index=t2d["Patient ID"])
    genders = genders.apply(lambda x: 1 if x == "Female" else 0).to_dict()
    ages = pd.Series(t2d["Age"].values, index=t2d["Patient ID"]).to_dict()
    hba1cs = pd.Series(t2d["HbA1c"].values, index=t2d["Patient ID"])

    bmis.dropna(how="any", inplace=True)
    ages.dropna(how="any", inplace=True)
    hba1cs.dropna(how="any", inplace=True)
    hba1cs = hba1cs.to_dict()

    # Reading demographic data for DirecNet patients, t1d
    _, t1d = read_direcnet_patient_info()
    direcnet_bmis = pd.Series(t1d["BMI"].values, index=t1d["PtID"]).to_dict()
    direcnet_ages = pd.Series(t1d["AgeAtBaseline"].values, index=t1d["PtID"]).to_dict()
    direcnet_hba1cs = pd.Series(t1d["HbA1c"].values, index=t1d["PtID"])
    direcnet_bmis.dropna(how="any", inplace=True)
    direcnet_ages.dropna(how="any", inplace=True)
    direcnet_hba1cs.dropna(how="any", inplace=True)
    direcnet_hba1cs = direcnet_hba1cs.to_dict()

    direcnet_genders = pd.Series(t1d["Gender"].values, index=t1d["PtID"])
    direcnet_genders = direcnet_genders.apply(lambda x: 1 if x == "F" else 0).to_dict()

    # store diabetic patients' data in all_data
    # because sometime feature is missing for some patients, so set feature
    # first

    t1d_feature = {
        k: list(np.tile(100 * direcnet_hba1cs[k], (len(low_fid_data.raw_data[k]), 1)))
        for k in direcnet_hba1cs.keys()
    }
    t2d_feature = {
        k: list(np.tile(100 * hba1cs[k], (len(high_fid_data.raw_data[k]), 1)))
        for k in hba1cs.keys()
    }
    t1d_data = {k: low_fid_data.raw_data[k] for k in direcnet_hba1cs}
    t2d_data = {k: high_fid_data.raw_data[k] for k in hba1cs}

    of = open("{}_{}.txt".format(label, cut_point), "a+")
    for item in t2d_feature.keys():
        print("Processing pid: {}".format(item))
        low_fid_data.data = sum(list(t1d_data.values()), [])
        low_fid_data.feature = sum(list(t1d_feature.values()), [])
        train_pids = set(t2d_feature.keys()) - set(item)
        for k in train_pids:
            low_fid_data.data += t2d_data[k]
            # set up feature vector
            low_fid_data.feature += t2d_feature[k]
        low_fid_data.reset(
            sampling_horizon, prediction_horizon, scale, 100, False, outtype, 1
        )

        high_fid_data.data = t2d_data[item]
        high_fid_data.set_cutpoint = cut_point
        high_fid_data.feature = t2d_feature[item]
        high_fid_data.reset(
            sampling_horizon,
            prediction_horizon,
            scale,
            train_test_ratio,
            False,
            outtype,
            1,
        )
        if cut_point > high_fid_data.train_n:
            print("Training data size smaller than required, skipped")
            continue
        with Pool(1) as p:
            err = p.apply(learner, (low_fid_data, high_fid_data, *argv))
            if transfer is not None:
                with Pool(1) as p:
                    terr1 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 1))
                    terr2 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 2))
                    terr3 = p.apply(transfer, (high_fid_data, argv[-3], argv[-2], 3))
                    of.write(
                        "{:s} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                            item, err, terr1, terr2, terr3
                        )
                    )
            else:
                of.write("{:s} {:.4f}\n".format(item, err))
    of.close()


if __name__ == "__main__":
    no_med, insulin, other = groupby_meds()
    with pd.ExcelWriter("CGMdata_meds.xlsx") as writer:
        no_med.to_excel(writer, sheet_name="no_med", index=False)
        insulin.to_excel(writer, sheet_name="insulin", index=False)
        other.to_excel(writer, sheet_name="other", index=False)
