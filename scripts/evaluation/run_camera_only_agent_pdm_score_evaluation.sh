TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/NAVSIM/exp/training_camera_only_agent/2025.04.16.07.21.52/camera_only/pjtrow5n/checkpoints/epoch49.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
metric_cache_path=$CACHE_PATH \
experiment_name=camera_only_agent