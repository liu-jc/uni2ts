import numpy as np
import torch
from chronos import ChronosPipeline
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

from uni2ts.eval_util.data import get_gluonts_test_dataset

# Load dataset
batch_size = 32
num_samples = 20
# dataset = get_dataset("m4_hourly")

# prediction_length = dataset.metadata.prediction_length

# Load Chronos
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

# Split dataset for evaluation
# _, test_template = split(dataset.test, offset=-prediction_length)
# test_data = test_template.generate_instances(prediction_length)

test_data, metadata = get_gluonts_test_dataset("m4_hourly")
prediction_length = metadata.prediction_length

# Generate forecast samples
forecast_samples = []
for batch in tqdm(batcher(test_data.input, batch_size=32)):
    context = [torch.tensor(entry["target"]) for entry in batch]
    forecast_samples.append(
        pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        ).numpy()
    )
forecast_samples = np.concatenate(forecast_samples)

# Convert forecast samples into gluonts SampleForecast objects
sample_forecasts = []
for item, ts in zip(forecast_samples, test_data.input):
    forecast_start_date = ts["start"] + len(ts["target"])
    sample_forecasts.append(
        SampleForecast(samples=item, start_date=forecast_start_date)
    )

# Evaluate
metrics_df = evaluate_forecasts(
    sample_forecasts,
    test_data=test_data,
    metrics=[
        MASE(),
        MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
    ],
)

print(metrics_df)
