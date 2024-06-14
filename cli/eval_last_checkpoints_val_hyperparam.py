# from cli import eval
import argparse
import itertools
import os
import time

import ipdb
import pandas as pd
from eval import evaluate
from hydra import compose, initialize
from omegaconf import OmegaConf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add an argument for "run_name"
    parser.add_argument("--run_name", type=str, help="Name of the run")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    args = parser.parse_args()

    print("start")
    initialize(version_base=None, config_path="conf/eval")
    dataset_names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
    # dataset_names = ['ETTh1', "ETTm1"]
    patch_szs = [64, 128]
    context_lengths = [x for x in range(2000, 5001, 1000)]
    time_dict = {}
    res_dict = {}
    dir_path = f"/export/lotsa-data/eval_outputs/eval_last_ckpt_results"
    checkpoint_dir_path = (
        f"/export/lotsa-data/outputs/pretrain/{args.model_name}/lotsa_v1_weighted/"
    )
    checkpoint_dir_path = os.path.join(
        checkpoint_dir_path, args.run_name, "checkpoints"
    )
    assert os.path.exists(checkpoint_dir_path), f"cannot find {checkpoint_dir_path}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    t1 = time.time()
    cols = ["pred_len", "variate", "psz", "ctx", "MSE[0.5]", "MAE[0.5]"]
    for dataset_name in dataset_names:
        print(f"----- Dataset Name: {dataset_name} -----")
        cur_path = os.path.join(
            dir_path,
            args.model_name,
            args.run_name,
            dataset_name,
            "val_hyperparam_results",
        )
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        checkpoint_file_path = os.path.join(checkpoint_dir_path, f"last.ckpt")
        tot_t1 = time.time()
        tot_time = 0
        result_list = []
        for pred_len in ["96", "192", "336", "720"]:
            print(f"----- pred length: {pred_len} -----")
            for mode in ["M", "S"]:
                print(f"--- mode: {mode} ---")
                for psz, ctx in itertools.product(patch_szs, context_lengths):
                    cur_log_file = os.path.join(
                        cur_path, f"psz{psz}" + f"ctx{ctx}" + f"-{mode}.log"
                    )
                    cur_t1 = time.time()
                    cfg = compose(
                        config_name="default",
                        overrides=[
                            "data=lsf_val",
                            f"data.dataset_name={dataset_name}",
                            f"data.mode={mode}",
                            f"patch_size={psz}",
                            'device="cuda"',
                            f"context_length={ctx}",
                            f"data.prediction_length={pred_len}",
                            "checkpoint_path=customized_checkpoint",
                            f"checkpoint_path.checkpoint_path='{checkpoint_file_path}'",
                        ],
                    )
                    res = evaluate(cfg)
                    if dataset_name not in res_dict:
                        time_dict[dataset_name] = {}
                        res_dict[dataset_name] = {}
                    if pred_len not in res_dict[dataset_name]:
                        res_dict[dataset_name][pred_len] = {}
                    res_dict[dataset_name][pred_len] = res
                    if res is None:
                        continue
                    print(
                        f'pred_len: {pred_len}, mode: {mode}, psz: {psz}, ctx: {ctx}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}'
                    )
                    result_list.append(
                        [
                            pred_len,
                            mode,
                            psz,
                            ctx,
                            res["MSE[0.5]"].item(),
                            res["MAE[0.5]"].item(),
                        ]
                    )
                    with open(cur_log_file, "a") as f:
                        f.write(
                            f'pred_len: {pred_len}, mode: {mode}, psz: {psz}, ctx: {ctx}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}\n'
                        )
        cur_df = pd.DataFrame(result_list, columns=cols)
        cur_df.to_csv(os.path.join(cur_path, "val_hyperparam-new.csv"), index=False)
        print("finished")
    print(f"Time spent: {time.time() - t1}s")
