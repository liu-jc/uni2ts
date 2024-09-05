import ipdb
from huggingface_hub import create_repo

from uni2ts.model.moirai.forecast import MoiraiForecast
from uni2ts.model.moirai.module import MoiraiModule

model = MoiraiForecast.load_from_checkpoint(
    # checkpoint_path='/export/lotsa-data/outputs/pretrain/moirai_base/lotsa_v1_all_data_univar/bs-256-10000-lr1e-3-bf16-mskrate0.15-0.5/checkpoints/last.ckpt',
    # checkpoint_path='/export/timeseries-data/shared_moirai_model/chliu-moirai1.1-cloudops1.0/moirai_base_2stage/cloudops_tsf/moirai_base/checkpoints/last.ckpt',
    # checkpoint_path='/export/lotsa-data/outputs/pretrain/moirai_small/only_synthdata_single/bs-256-1000-lr1e-3-mskrate0.15-0.5/checkpoints/last-v1.ckpt',
    # checkpoint_path='/export/lotsa-data/outputs/pretrain/moirai_base/only_synthdata_single/bs-256-10000-lr1e-3-bf16-mskrate0.15-0.5/checkpoints/last.ckpt',
    # checkpoint_path="/export/timeseries-data/chliu/uni2ts-branches/downsample/uni2ts-sfr/outputs/pretrain/moirai_small_2stage/cloudops_tsf/moirai_small/checkpoints/last.ckpt",
    checkpoint_path="/export/lotsa-data/outputs/pretrain/moirai_small/tsmixup_10m/bs-256-1000-lr1e-3-mskrate0.15-0.5/checkpoints/last.ckpt",
    prediction_length=100,
    context_length=1200,
    patch_size=32,
    num_samples=100,
    target_dim=2,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)
model.module.push_to_hub("juncliu/moirai_1.1_R_small_tsmixup_10m")
# ipdb.set_trace()
