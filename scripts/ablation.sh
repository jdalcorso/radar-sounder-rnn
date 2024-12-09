#!/bin/bash
script_dir=$1
log_dir=$2
container_name=$3

train_script=train
test_script=test

# Clear previous (or create) experiment file
echo -n > $log_dir/latest_ablation.txt

# Whether to manually set model paramters
model='nlur'
sed -i "s|model: .*|model: $model|g" $script_dir/config_files/$train_script.yaml
sed -i "s|model: .*|model: $model|g" $script_dir/config_files/$test_script.yaml

# Multi-seed experiment
for PATCH in 16 32 64
do
    for SEQ in 4 8 16 32 64
    do
        docker restart $container_name
        sed -i "s|seq_len: .*|seq_len: $SEQ|g" $script_dir/config_files/$train_script.yaml
        sed -i "s|seq_len: .*|seq_len: $SEQ|g" $script_dir/config_files/$test_script.yaml
        sed -i "s|patch_len: .*|patch_len: $PATCH|g" $script_dir/config_files/$train_script.yaml
        sed -i "s|patch_len: .*|patch_len: $PATCH|g" $script_dir/config_files/$test_script.yaml
        docker exec $container_name python $script_dir/$train_script.py -c $script_dir/config_files/$train_script.yaml
        docker exec $container_name python $script_dir/$test_script.py -c $script_dir/config_files/$test_script.yaml >> $log_dir/latest_ablation.txt
    done
done

# Reset
sed -i "s|seq_len: .*|seq_len: $SEQ|g" $script_dir/config_files/$train_script.yaml
sed -i "s|seq_len: .*|seq_len: $SEQ|g" $script_dir/config_files/$test_script.yaml
sed -i "s|patch_len: .*|patch_len: $PATCH|g" $script_dir/config_files/$train_script.yaml
sed -i "s|patch_len: .*|patch_len: $PATCH|g" $script_dir/config_files/$test_script.yaml