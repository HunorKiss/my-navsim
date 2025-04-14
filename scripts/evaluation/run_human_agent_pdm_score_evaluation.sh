TRAIN_TEST_SPLIT=navtest
CACHE_PATH=YOUR_PATH_TO_METRIC_CACHE

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=camera_only_agent
