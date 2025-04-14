TEAM_NAME="KHTeam"
AUTHORS="Kiss_Hunor"
EMAIL="kisshunor@edu.bme.hu"
INSTITUTION="Budapest_University_of_Technology_and_Economics"
COUNTRY="Hungary"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/root/workdir/NAVSIM/exp/training_camera_only_agent/2025.04.01.16.17.10/lightning_logs/version_0/checkpoints/epoch13.ckpt


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
