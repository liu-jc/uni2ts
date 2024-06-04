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

from typing import Any, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.distributions import Distribution

from ._base import PackedDistributionLoss


class PackedNLLLoss(PackedDistributionLoss):
    def __call__(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]] = None,
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        return_total_loss: bool = False,
    ) -> tuple[Tensor, Tensor] | Tensor:
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        total_loss = self._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        reduced_loss = self.reduce_loss(
            total_loss, prediction_mask, observed_mask, sample_id, variate_id
        )
        if return_total_loss:
            return total_loss, reduced_loss
        else:
            return reduced_loss

    def _loss_func(
        self,
        pred: Distribution,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return -pred.log_prob(target)

    def _loss_per_ds(
        self,
        batched_samples: dict[str, Any],
        loss_per_ele: Float[torch.Tensor, "*batch seq_len #dim"],
        field: str,
    ):
        loss_per_ds = {}
        loss_per_patch = loss_per_ele.mean(-1)  # average over the hidden dimension
        # print('loss_per_patch.size():', loss_per_patch.size())

        # Flatten the array
        assert field in batched_samples
        array = batched_samples[field]
        flat_array = array.flatten()

        # Find unique strings and their first occurrence indices in the flattened array
        unique_strings, first_occurrence_indices = np.unique(
            flat_array, return_index=True
        )

        # Create a dictionary to map unique strings to their indices in the original array
        unique_indices_dict = {}

        for string in unique_strings:
            indices = np.argwhere(array == string)
            unique_indices_dict[string] = indices
            rows, cols = indices[:, 0], indices[:, 1]
            loss_per_ds[string] = torch.mean(loss_per_patch[rows, cols].detach())

        return loss_per_ds
