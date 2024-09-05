#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

import hydra
import torch
from gluonts.time_feature import get_seasonality
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.eval_util.evaluation import evaluate_model


def evaluate(cfg):
    # print(OmegaConf.to_yaml(cfg))
    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size
    print(cfg.checkpoint_path)
    while True:
        if callable(cfg.checkpoint_path):
            checkpoint_path = call(cfg.checkpoint_path)
        else:
            checkpoint_path = cfg.checkpoint_path.checkpoint_path
        model = call(cfg.load_from_checkpoint)(
            prediction_length=metadata.prediction_length,
            # checkpoint_path=call(cfg.checkpoint_path),
            # checkpoint_path=cfg.checkpoint_path.checkpoint_path,
            checkpoint_path=checkpoint_path,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
            context_length=cfg.context_length,
            patch_size=cfg.patch_size,
        )
        metrics = instantiate(cfg.metrics, _convert_="all")
        try:
            predictor = model.create_predictor(batch_size, cfg.device)
            res = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=cfg.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(metadata.freq),
            )
            print(res)
            return res
            # output_dir = HydraConfig.get().runtime.output_dir
            # writer = SummaryWriter(
            #     log_dir=os.path.join(
            #         output_dir,
            #         f"patch_size={cfg.patch_size}",
            #         f"context_length={cfg.context_length}",
            #     )
            # )
            # for name, metric in res.to_dict("records")[0].items():
            #     writer.add_scalar(f"{metadata.split}_metrics/{name}", metric)
            # writer.close()
            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2
            if batch_size < cfg.min_batch_size:
                print(
                    f"batch_size {batch_size} smaller than min_batch_size {cfg.min_batch_size}, "
                    f"ending evaluation of patch_size {cfg.patch_size}, context_length {cfg.context_length}"
                )
                break


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size
    while True:
        model = call(cfg.model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )
        metrics = instantiate(cfg.metrics, _convert_="all")
        print(
            "-" * 5,
            f"Evaluating {cfg.data.dataset_name} with model_path {cfg.model.module.pretrained_model_name_or_path}",
            "-" * 5,
        )
        try:
            predictor = model.create_predictor(batch_size, cfg.device)
            res = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=cfg.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(metadata.freq),
            )
            res.index = [cfg.data.dataset_name]
            print(res)
            output_dir = HydraConfig.get().runtime.output_dir
            # print('output_dir:', output_dir)
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics/{name}", metric)
            writer.close()

            results_dir = os.path.join(cfg.results_dir, cfg.run_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            save_path = os.path.join(results_dir, f"{cfg.data.dataset_name}.csv")
            res.to_csv(save_path)
            print(f"Results saved to {save_path}")
            print("-" * 5, f"Evaluation on {cfg.data.dataset_name} completed", "-" * 5)

            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2
            if batch_size < cfg.min_batch_size:
                print(
                    f"batch_size {batch_size} smaller than "
                    f"min_batch_size {cfg.min_batch_size}, ending evaluation"
                )
                break


if __name__ == "__main__":
    main()
