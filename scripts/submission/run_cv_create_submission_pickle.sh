TEAM_NAME="ContiBME"
AUTHORS="Hunor Kiss"
EMAIL="kisshunor@edu.bme.hu"
INSTITUTION="Budapest University of Technology and Economics"
COUNTRY="Hungary"

TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=/root/workdir/NAVSIM/exp/training_camera_only_agent/2025.04.27.10.16.13/camera_only/mir25tuj/checkpoints/epoch12.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_challenge.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=submission_camera_only_agent \
agent.checkpoint_path=$CHECKPOINT \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
