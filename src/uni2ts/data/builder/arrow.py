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

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import datasets
import ipdb
import pandas as pd

# from gluonts.dataset.common import FileDataset
import pyarrow as pa
from datasets import Features, Sequence, Value, load_dataset
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import EvalDataset, SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation

from ._base import DatasetBuilder


@dataclass
class ArrowDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        dataset: str,
        weight: float = 1.0,
        sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE,
        storage_path: Path = env.CUSTOM_DATA_PATH,
        use_stream: bool = False,
    ):
        self.dataset = dataset
        self.weight = weight
        self.sample_time_series = sample_time_series
        self.storage_path = storage_path
        self.storage_path = Path(self.storage_path)
        self.file_name = self.dataset + ".arrow"
        self.use_stream = use_stream

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)
        self.file_name = self.dataset + ".arrow"
        # print(self.storage_path / self.file_name)
        # print(self.ds)
        # ipdb.set_trace()

    def build_dataset(self, dataset: str):
        def gen_func() -> Generator[dict[str, Any], None, None]:
            if self.use_stream:
                # table = pa.ipc.open_stream(self.storage_path / self.file_name).read_all()
                dataset = load_dataset(
                    "arrow",
                    data_files=str(self.storage_path / self.file_name),
                    split="train",
                    streaming=True,
                )
                for index, row in enumerate(dataset):
                    yield dict(
                        item_id=index,
                        start=row["start"],
                        target=row["target"],
                        freq="h",
                    )
            else:
                table = pa.ipc.open_file(self.storage_path / self.file_name).read_all()
                df = table.to_pandas()
                freq = "h"
                for index, row in df.iterrows():
                    yield dict(
                        item_id=index,
                        start=row["start"],
                        target=row["target"],
                        freq=freq,
                    )

        hf_dataset = datasets.Dataset.from_generator(
            generator=gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                )
            ),
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(dataset_path=env.CUSTOM_DATA_PATH / dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.Dataset.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
            dataset_name=self.dataset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument(
        "--use_stream",
        action="store_true",
        help="Indicates whether to use stream files",
    )
    args = parser.parse_args()

    ds = ArrowDatasetBuilder(
        dataset=args.dataset_name, use_stream=args.use_stream
    ).build_dataset(args.dataset_name)
