# from cli import eval
import argparse
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
    parser.add_argument("--max_epochs", type=int, help="max epochs")
    args = parser.parse_args()

    print("start")
    initialize(version_base=None, config_path="conf/eval")
    # dataset_names =  ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather"]
    dataset_names = ["ETTh1"]
    patch_sz = 64
    context_length = 5000
    time_dict = {}
    res_dict = {}
    dir_path = f"/export/lotsa-data/eval_outputs/ckpts_eval_results"
    checkpoint_dir_path = (
        f"/export/lotsa-data/outputs/pretrain/{args.model_name}/lotsa_v1_weighted/"
    )
    checkpoint_dir_path = os.path.join(
        checkpoint_dir_path, args.run_name, "checkpoints"
    )
    assert os.path.exists(checkpoint_dir_path), f"cannot find {checkpoint_dir_path}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for dataset_name in dataset_names:
        print(f"----- Dataset Name: {dataset_name} -----")
        cur_path = os.path.join(
            dir_path, args.model_name, args.run_name, dataset_name, "val"
        )
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        cur_log_file = os.path.join(
            cur_path, f"psz{patch_sz}" + f"ctx{context_length}" + "-S.log"
        )
        cols = ["epoch", "pred_len", "MSE[0.5]", "MAE[0.5]"]
        result_list = []
        for epoch in range(50, args.max_epochs + 1, 50):
            # cur_log_file = open(os.path.join(dir_path, dataset_name + "-S.log"), "a")
            print(f"---epoch: {epoch}---")
            checkpoint_file_path = os.path.join(
                checkpoint_dir_path, f"epoch={epoch-1}-step={epoch*100}.ckpt"
            )
            tot_t1 = time.time()
            tot_time = 0
            for pred_len in ["96", "192", "336", "720"]:
                print(f"----- pred length: {pred_len} -----")
                cur_t1 = time.time()
                cfg = compose(
                    config_name="default",
                    overrides=[
                        "data=lsf_val",
                        f"data.dataset_name={dataset_name}",
                        "data.mode=S",
                        f"patch_size={patch_sz}",
                        'device="cuda"',
                        f"context_length={context_length}",
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
                res_dict[dataset_name][pred_len][epoch] = res
                if res is None:
                    continue
                result_list.append(
                    [epoch, pred_len, res["MSE[0.5]"].item(), res["MAE[0.5]"].item()]
                )
                print(
                    f'epoch: {epoch}, pred_len: {pred_len}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}'
                )
                with open(cur_log_file, "a") as f:
                    f.write(
                        f'epoch: {epoch}, pred_len: {pred_len}, MSE[0.5]: {res["MSE[0.5]"].item()}, MAE[0.5]: {res["MAE[0.5]"].item()}\n'
                    )
        cur_df = pd.DataFrame(result_list, columns=cols)
        cur_df.to_csv(os.path.join(cur_path, "all_eval_results.csv"), index=False)
        print("finished")
