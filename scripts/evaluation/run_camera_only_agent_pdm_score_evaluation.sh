TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=/root/workdir/NAVSIM/exp/coa_dinovits8_addition/2025.05.06.22.11.08/camera_only/799a9cch/checkpoints/epoch35.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=coa_dinovits8_addition_evaluation \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
experiment_name=camera_only_agent