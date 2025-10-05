TRAIN_TEST_SPLIT=navtrain
EXPERIMENT_NAME=training_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
worker=single_machine_thread_pool \
experiment_name=$EXPERIMENT_NAME