TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=$NAVSIM_EXP_ROOT/auxiliary_model_w_agents/2025.10.04.13.34.17/thesis/t8p3gwa1/checkpoints/epoch39.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=auxiliary_model_w_agents_evaluation \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH