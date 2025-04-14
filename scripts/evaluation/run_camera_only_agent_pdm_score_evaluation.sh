TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/root/workdir/NAVSIM/exp/training_camera_only_agent/2025.04.01.16.17.10/lightning_logs/version_0/checkpoints/epoch13.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=camera_only_agent