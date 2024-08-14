#for ds in us_births saugeenday sunspot_with_missing temperature_rain_with_missing covid_deaths hospital rideshare_with_missing traffic_weekly traffic_hourly fred_md car_parts_with_missing electricity_weekly electricity_hourly solar_weekly solar_10_minutes nn5_weekly nn5_daily_with_missing weather kdd_cup_2018_with_missing vehicle_trips_with_missing pedestrian_counts bitcoin_with_missing dominick australian_electricity_demand cif_2016_12 cif_2016_6 tourism_monthly tourism_quarterly m4_hourly m4_daily m4_weekly m4_monthly monash_m3_other monash_m3_monthly m1_monthly m1_yearly monash_m3_yearly m4_yearly tourism_yearly m1_quarterly monash_m3_quarterly m4_quarterly kaggle_web_traffic_weekly kaggle_web_traffic_with_missing bitcoin car_parts_without_missing nn5_daily_without_missing kaggle_web_traffic_without_missing vehicle_trips_without_missing temperature_rain_without_missing sunspot_without_missing rideshare_without_missing kdd_cup_2018_without_missing
#do
#  python -m cli.eval \
#    run_name=moirai_eval_small_univar \
#    results_dir=./project/benchmarks/results/ \
#    model=moirai_1.1_R_small_univar \
#    model.patch_size=32 \
#    model.context_length=1000 \
#    data=monash \
#    data.dataset_name=$ds
#done



model_name=moirai_1.1_R_base_all_univar_data
run_name=moirai_base_all_univar_data_200B
for ds in m1_yearly monash_m3_yearly m4_yearly tourism_yearly m1_quarterly monash_m3_quarterly m4_quarterly tourism_quarterly; do
  python -m cli.eval \
    run_name=${run_name} \
    results_dir=./project/benchmarks/results/ \
    model=${model_name} \
    model.patch_size=8 \
    model.context_length=1000 \
    data=monash \
    data.dataset_name=$ds
done;
for ds in m1_monthly monash_m3_monthly monash_m3_other m4_monthly tourism_monthly cif_2016_6 cif_2016_12 car_parts_with_missing car_parts_without_missing fred_md hospital m4_weekly nn5_weekly kaggle_web_traffic_weekly solar_weekly electricity_weekly dominick traffic_weekly m4_daily nn5_daily_with_missing nn5_daily_without_missing kaggle_web_traffic_with_missing kaggle_web_traffic_without_missing bitcoin vehicle_trips_with_missing vehicle_trips_without_missing covid_deaths weather temperature_rain_with_missing temperature_rain_without_missing sunspot_with_missing sunspot_without_missing saugeenday us_births; do
  python -m cli.eval \
    run_name=${run_name} \
    results_dir=./project/benchmarks/results/ \
    model=${model_name} \
    model.patch_size=32 \
    model.context_length=1000 \
    data=monash \
    data.dataset_name=$ds
done;
for ds in m4_hourly electricity_hourly traffic_hourly pedestrian_counts rideshare_with_missing rideshare_without_missing kdd_cup_2018_with_missing kdd_cup_2018_without_missing australian_electricity_demand; do
  python -m cli.eval \
    run_name=${run_name} \
    results_dir=./project/benchmarks/results/ \
    model=${model_name} \
    model.patch_size=64 \
    model.context_length=1000 \
    data=monash \
    data.dataset_name=$ds
done;