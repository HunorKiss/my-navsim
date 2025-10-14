export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT=~/thesis/exp
export NAVSIM_DEVKIT_ROOT=~/thesis/my-navsim
export OPENSCENE_DATA_ROOT=~/thesis/dataset
export NUPLAN_MAPS_ROOT=~/thesis/dataset/maps
export PYTHONPATH=$NAVSIM_DEVKIT_ROOT:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0

TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=camera_only_agent \
experiment_name=training_model_alpha \
trainer.params.max_epochs=50 \
train_test_split=$TRAIN_TEST_SPLIT