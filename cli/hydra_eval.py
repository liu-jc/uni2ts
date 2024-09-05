# from cli import eval
import os
import time

import ipdb
from eval import evaluate
from hydra import compose, initialize
from omegaconf import OmegaConf

if __name__ == "__main__":
    print("start")
    initialize(version_base=None, config_path="conf/eval")
    dataset_names = ["ETTh1"]
    patch_sz = {
        "ETTh1": 64,
        "ETTh2": 64,
        "ETTm1": 128,
        "ETTm2": 64,
        "electricity": 64,
        "weather": 128,
    }
    context_lengths = {
        "ETTh1": 5000,
        "ETTh2": 3000,
        "ETTm1": 4000,
        "ETTm2": 3000,
        "electricity": 5000,
        "weather": 2000,
    }
    time_dict = {}
    res_dict = {}
    dir_path = "./eval_results"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for dataset_name in dataset_names:
        print(f"----- Dataset Name: {dataset_name} -----")
        cur_log_file = open(os.path.join(dir_path, dataset_name + "-S.log"), "a")
        tot_t1 = time.time()
        tot_time = 0
        for pred_len in ["96", "192", "336", "720"]:
            print(f"----- pred length: {pred_len} -----")
            cur_t1 = time.time()
            cfg = compose(
                config_name="default",
                overrides=[
                    "data=lsf_test",
                    f"data.dataset_name={dataset_name}",
                    "data.mode=S",
                    f"patch_size={patch_sz[dataset_name]}",
                    'device="cuda"',
                    f"context_length={context_lengths[dataset_name]}",
                    f"data.prediction_length={pred_len}",
                    "checkpoint_path=moirai_R_small",
                ],
            )
            res = evaluate(cfg)
            if dataset_name not in time_dict:
                time_dict[dataset_name] = {}
                res_dict[dataset_name] = {}
            time_dict[dataset_name][pred_len] = time.time() - cur_t1
            tot_time += time_dict[dataset_name][pred_len]
            res_dict[dataset_name][pred_len] = res
            if res is None:
                continue
            print(
                f'pred_len: {pred_len}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}'
            )
            cur_log_file.write(
                f'pred_len: {pred_len}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}\n'
            )
        tot_endt = time.time()
        time_dict[dataset_name]["total"] = tot_time
        print(f"Total Time elapsed for {dataset_name}: {tot_endt-tot_t1}s")
        print(f"individual time for {dataset_name}")
        print(time_dict[dataset_name])
    for dataset_name in dataset_names:
        print(f"individual time for {dataset_name}")
        print(time_dict[dataset_name])
