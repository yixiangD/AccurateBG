import json
import os
import subprocess


def update_sh(folder=0, ph=6):
    of = open("./mpi.sh", "r")
    text = of.read()
    of.close()
    text += f"python3 ohio_main.py --epoch 200 --outdir ../{folder} --prediction_horizon {ph}\n"
    # print(text)
    of = open("accurate_bg/mpi.sh", "w")
    of.write(text)
    of.close()


def main():
    config = {}
    config["batch_size"] = 64
    config["learning_rate"] = 1e-2
    # default parameters
    config["k_size"] = 4
    config["nblock"] = 4
    config["nn_size"] = 10
    config["nn_layer"] = 2
    config["beta"] = 1e-4
    config["loss"] = "rmse"
    folder = "output"
    with open(f"./{folder}/config.json", "w") as outfile:
        json.dump(config, outfile)
    update_sh(folder, 12)
    subprocess.check_call("cd accurate_bg && sbatch mpi.sh && cd ../", shell=True)
    update_sh(folder)
    subprocess.check_call("cd accurate_bg && sbatch mpi.sh && cd ../", shell=True)
    exit()
    p = 0
    for k in [2, 4, 6]:
        for n in [2, 4, 8]:
            for s in [4, 8, 16]:
                for nl in [2, 4, 8]:
                    for bt in [1e-2, 1e-3, 1e-4]:
                        folder = f"test_{p}"
                        os.mkdir(folder)
                        config["k_size"] = k
                        config["nblock"] = n
                        config["nn_size"] = s
                        config["nn_layer"] = nl
                        config["beta"] = bt
                        path = os.path.join(folder, "config.json")
                        with open(path, "w") as outfile:
                            json.dump(config, outfile)
                        p += 1
                        update_sh(folder)
                        subprocess.check_call(
                            "cd accurate_bg && sbatch mpi.sh && cd ../", shell=True
                        )


if __name__ == "__main__":
    main()
