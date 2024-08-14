#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <size>"
    echo "Allowed sizes: small, tiny, mini, base"
    exit 1
fi

# List of allowed sizes
allowed_sizes=("small" "tiny" "mini" "base")

# Check if the provided size is in the allowed list
if ! [[ "${allowed_sizes[@]}" =~ "$1" ]]; then
    echo "Error: Size must be one of: ${allowed_sizes[*]}"
    exit 1
fi

model_size=$1
# Print out the size argument passed to the script
echo "Model Size selected: ${model_size}"
model_path=/export/home/chronos-forecasting/checkpoints/chronos-${model_size}_univar_data_w_synthdata
for ds in us_births saugeenday sunspot_with_missing temperature_rain_with_missing covid_deaths hospital rideshare_with_missing traffic_weekly traffic_hourly fred_md car_parts_with_missing electricity_weekly electricity_hourly solar_weekly solar_10_minutes nn5_weekly nn5_daily_with_missing weather kdd_cup_2018_with_missing vehicle_trips_with_missing pedestrian_counts bitcoin_with_missing dominick australian_electricity_demand cif_2016_12 cif_2016_6 tourism_monthly tourism_quarterly m4_hourly m4_daily m4_weekly m4_monthly monash_m3_other monash_m3_monthly m1_monthly m1_yearly monash_m3_yearly m4_yearly tourism_yearly m1_quarterly monash_m3_quarterly m4_quarterly kaggle_web_traffic_weekly kaggle_web_traffic_with_missing bitcoin car_parts_without_missing nn5_daily_without_missing kaggle_web_traffic_without_missing vehicle_trips_without_missing temperature_rain_without_missing sunspot_without_missing rideshare_without_missing kdd_cup_2018_without_missing
#for ds in car_parts_without_missing nn5_daily_without_missing kaggle_web_traffic_without_missing vehicle_trips_without_missing temperature_rain_without_missing sunspot_without_missing rideshare_without_missing kdd_cup_2018_without_missing
do
  python run_chronos.py --model_path=${model_path} --dataset=${ds}  --run_name=chronos-${model_size}_univar_data_w_synthdata
done