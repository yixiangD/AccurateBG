from __future__ import division, print_function

import collections
import csv
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


class DataReader(object):
    """
    Read continuous glucose data
    """

    def __init__(self, fmt, filepath, sampling_interval):
        """
        fmt: string
        filepath: string, path of file
        sampling_interval: float, int or datetime.timedelta, in minutes
        """
        if fmt not in ["direcnet", "VA", "VA1", "VA2", "direcnet_pid", "ohio"]:
            raise ValueError("Wrong data format")
        self.fmt = fmt
        self.filepath = filepath
        if type(sampling_interval) is datetime.timedelta:
            self.interval = sampling_interval.seconds / 60
            self.interval_timedelta = sampling_interval
        elif type(sampling_interval) in [float, int]:
            self.interval = sampling_interval
            self.interval_timedelta = datetime.timedelta(minutes=sampling_interval)

    def read(self):
        """
        return a list of list, where each sublist is the glucose history patients every 5-min
        """
        if self.fmt == "direcnet":
            return self.read_direcnet()
        elif self.fmt == "VA":
            return self.read_VA()
        elif self.fmt == "VA1":
            return self.read_VA_patient1()
        elif self.fmt == "VA2":
            return self.read_VA_patient2()
        elif self.fmt == "direcnet_pid":
            return self.read_direcnet_pid()
        elif self.fmt == "ohio":
            return self.read_ohio()

    def read_direcnet(self):
        """
        http://direcnet.jaeb.org/Studies.aspx?RecID=155
        tblADataCGMS.csv
        """
        data = collections.defaultdict(list)
        with open(self.filepath, "r") as csvfile:
            f = csv.DictReader(csvfile)
            for row in f:
                t = datetime.datetime.strptime(
                    row["ReadingDt"][:11] + row["ReadingTm"], "%Y-%m-%d %I:%M %p"
                )
                data[int(row["PtID"])].append((t, float(row["SensorGLU"])))

        bag = []
        zero_seconds = datetime.timedelta()
        for d in data.values():
            d.sort(key=lambda t: t[0])
            bag.append([d[0]])
            for i in range(1, len(d)):
                dt = d[i][0] - bag[-1][-1][0]
                if dt == self.interval_timedelta:
                    bag[-1].append(d[i])
                elif dt > self.interval_timedelta:
                    bag.append([d[i]])
                elif dt == zero_seconds:
                    bag[-1][-1] = (d[i][0], (bag[-1][-1][1] + d[i][1]) / 2)
                else:
                    if i + 1 < len(d) and d[i + 1][0] - bag[-1][-1][0] == self.interval:
                        continue
                    bag.append([d[i]])
        return list(map(lambda l: np.array(list(zip(*l))[1]), bag))

    def read_direcnet_pid(self):
        """
        read direcnet patients blood glucose and return a dictionary
        """
        df = pd.read_csv(self.filepath)
        date = df["ReadingDt"].apply(lambda x: x.split()[0] + " ")
        subdf = df[["PtID", "SensorGLU"]]
        time = date + df["ReadingTm"]
        time = time.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %I:%M %p"))
        time = pd.DataFrame(time, columns=["Time"])
        newdf = pd.concat([subdf, time], axis=1)

        grouped = newdf.groupby("PtID")
        data = [grouped.get_group(x).reset_index(drop=True) for x in grouped.groups]

        res = collections.defaultdict(list)
        for item in data:
            pid = item["PtID"][0]
            res[pid].append([float(item["SensorGLU"][0])])
            for i in range(1, len(item.index)):
                delt = item["Time"][i] - item["Time"][i - 1]
                if delt <= self.interval_timedelta:
                    res[pid][-1].append(float(item["SensorGLU"][i]))
                else:
                    res[pid].append([float(item["SensorGLU"][i])])
            res[pid] = list(map(lambda d: np.array(d), res[pid]))
        return res

    def read_VA_patient(self, filepath):
        """
        VA hospital: Prof Mantzoros
        possible rows:
            - 1 2000-01-01T23:59:00 EGV                 SM00000000 100  ...
            - 1 2000-01-01T23:59:00 EGV         High    SM00000000 High ...
            - 1 2000-01-01T23:59:00 EGV         Low     SM00000000 Low  ...
            - 1 2000-01-01T23:59:00 Calibration         SM00000000 100
        """
        interval_eps = datetime.timedelta(seconds=5)

        t0 = datetime.datetime.min
        data = []
        with open(filepath, "r") as fin:
            for line in fin:
                contents = line.split("\t")
                if contents[2] == "Calibration":
                    continue
                t = datetime.datetime.strptime(contents[1], "%Y-%m-%dT%H:%M:%S")
                if abs(t - t0 - self.interval_timedelta) > interval_eps:
                    data.append([])
                data[-1].append(
                    400
                    if contents[7] == "High"
                    else 40
                    if contents[7] == "Low"
                    else float(contents[7])
                )
                t0 = t
        return list(map(lambda d: np.array(d), data))

    def read_VA_patient1(self):
        """
        VA hospital new: Prof Mantzoros
        """
        rawdata = pd.read_excel(self.filepath)
        data = pd.concat(
            [rawdata["Date"], rawdata["Time"], rawdata["Sensor Glucose"]], axis=1
        )
        data = data.dropna()
        time = pd.to_datetime(data.Date.astype(str) + " " + data.Time.astype(str))
        time.name = "Time"
        data = pd.concat([time, data["Sensor Glucose"]], axis=1)

        interval_eps = datetime.timedelta(seconds=5)

        output = []
        output.append([data["Sensor Glucose"].iloc[0]])
        t0 = data["Time"].iloc[0]
        for i in data.index[1:]:
            if abs(data["Time"][i] - t0 - self.interval_timedelta) > interval_eps:
                output.append([])
            output[-1].append(data["Sensor Glucose"][i])
            t0 = data["Time"][i]
        return list(map(lambda d: np.array(d), output))

    def read_VA_patient2(self):
        """
        VA hospital add new: Prof.Mantzoros, no 'Date' provided
        """
        excel = pd.ExcelFile(self.filepath)
        sheets = []
        for name in excel.sheet_names:
            df = excel.parse(name)
            if "Sensor Glucose" in df.columns:
                subdf = pd.concat(
                    [df["Patient ID"], df["Time"], df["Sensor Glucose"]], axis=1
                )
                sheets.append(subdf)
        all_data = pd.concat(sheets, axis=0)
        # print(all_data.describe())
        grouped = all_data.groupby("Patient ID")
        data = [grouped.get_group(x).reset_index(drop=True) for x in grouped.groups]

        res = collections.defaultdict(list)
        for item in data:
            pid = item["Patient ID"][0]
            res[pid].append([float(item["Sensor Glucose"][0])])
            for i in range(1, len(item.index)):
                t1 = datetime.datetime.combine(datetime.date.min, item["Time"][i])
                t0 = datetime.datetime.combine(datetime.date.min, item["Time"][i - 1])
                delt = t1 - t0
                if delt <= self.interval_timedelta:
                    res[pid][-1].append(float(item["Sensor Glucose"][i]))
                else:
                    res[pid].append([float(item["Sensor Glucose"][i])])
            res[pid] = list(map(lambda d: np.array(d), res[pid]))
        return res

    def read_VA(self):
        data = []
        for filepath in self.filepath:
            data += self.read_VA_patient(filepath)
        return data

    def read_ohio(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        res = []
        for item in root.findall("glucose_level"):
            entry0 = item[0].attrib
            res.append([float(entry0["value"])])
            for i in range(1, len(item)):
                last_entry = item[i - 1].attrib
                entry = item[i].attrib
                t1 = datetime.datetime.strptime(entry["ts"], "%d-%m-%Y %H:%M:%S")
                t0 = datetime.datetime.strptime(last_entry["ts"], "%d-%m-%Y %H:%M:%S")
                delt = t1 - t0
                if delt <= self.interval_timedelta:
                    res[-1].append(float(entry["value"]))
                else:
                    res.append([float(entry["value"])])
        return res
