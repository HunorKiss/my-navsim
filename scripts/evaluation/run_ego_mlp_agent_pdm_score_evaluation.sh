TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/root/workdir/NAVSIM/exp/training_ego_mlp_agent/2025.03.25.20.52.31/lightning_logs/version_0/checkpoints/ep49.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
