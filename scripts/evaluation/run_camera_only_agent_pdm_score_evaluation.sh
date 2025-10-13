export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=~/thesis/exp
export NAVSIM_DEVKIT_ROOT=~/thesis/my-navsim
export OPENSCENE_DATA_ROOT=~/thesis/dataset
export NUPLAN_MAPS_ROOT=~/thesis/dataset/maps
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1 

TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=$NAVSIM_EXP_ROOT/complex_model_w_agent_bev_v2/2025.10.12.13.30.39/thesis/cv8mvjd5/checkpoints/epoch14.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=eval_complex_model_w_agent_bev_v2 \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH