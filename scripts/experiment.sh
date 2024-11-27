#!/bin/bash
script_dir=$1
log_dir=$2
container_name=$3
train_script=$4

# Set test script
if [ "$train_script" == "train" ]; then
    test_script="test"
else
    test_script="test_weak"
fi

# Clear previous (or create) experiment file
echo -n > $log_dir/latest_experiment.txt

# Whether to manually set model paramters
model='nlur'
sed -i "s|model: .*|model: $model|g" $script_dir/config_files/$train_script.yaml
sed -i "s|model: .*|model: $model|g" $script_dir/config_files/$test_script.yaml

# Multi-seed experiment
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    docker restart $container_name
    sed -i "s|seed: .*|seed: $SEED|g" $script_dir/config_files/$train_script.yaml
    sed -i "s|seed: .*|seed: $SEED|g" $script_dir/config_files/$test_script.yaml
    docker exec $container_name python $script_dir/$train_script.py -c $script_dir/config_files/$train_script.yaml
    docker exec $container_name python $script_dir/$test_script.py -c $script_dir/config_files/$test_script.yaml >> $log_dir/latest_experiment.txt
done

# Reset seed
sed -i "s|seed: .*|seed: 0|g" $script_dir/config_files/$train_script.yaml
sed -i "s|seed: .*|seed: 0|g" $script_dir/config_files/$test_script.yaml

# Process results
docker exec $container_name python $script_dir/process_results.py -f $log_dir/latest_experiment.txt >> $log_dir/latest_experiment.txt
