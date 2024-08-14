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

from ._base import LOTSADatasetBuilder
from .buildings_bench import Buildings900KDatasetBuilder, BuildingsBenchDatasetBuilder
from .cloudops_tsf import CloudOpsTSFDatasetBuilder, CloudOpsTSFUnivarDatasetBuilder
from .cmip6 import CMIP6DatasetBuilder, CMIP6UnivarDatasetBuilder
from .era5 import ERA5DatasetBuilder, ERA5UnivarDatasetBuilder
from .gluonts import GluonTSDatasetBuilder
from .largest import LargeSTDatasetBuilder
from .lib_city import LibCityDatasetBuilder, LibCityUnivarDatasetBuilder
from .others import OthersLOTSADatasetBuilder, OthersUnivarLOTSADatasetBuilder
from .proenfo import ProEnFoDatasetBuilder
from .subseasonal import SubseasonalDatasetBuilder, SubseasonalUnivarDatasetBuilder

__all__ = [
    "LOTSADatasetBuilder",
    "Buildings900KDatasetBuilder",
    "BuildingsBenchDatasetBuilder",
    "CloudOpsTSFDatasetBuilder",
    "CloudOpsTSFUnivarDatasetBuilder",
    "CMIP6DatasetBuilder",
    "CMIP6UnivarDatasetBuilder",
    "ERA5DatasetBuilder",
    "ERA5UnivarDatasetBuilder",
    "GluonTSDatasetBuilder",
    "LargeSTDatasetBuilder",
    "LibCityDatasetBuilder",
    "LibCityUnivarDatasetBuilder",
    "OthersLOTSADatasetBuilder",
    "OthersUnivarLOTSADatasetBuilder",
    "ProEnFoDatasetBuilder",
    "SubseasonalDatasetBuilder",
    "SubseasonalUnivarDatasetBuilder",
]
