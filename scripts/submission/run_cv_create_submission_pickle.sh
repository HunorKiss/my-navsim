TEAM_NAME="ContiBME_KH"
AUTHORS="Hunor_Kiss"
EMAIL="kisshunor@edu.bme.hu"
INSTITUTION="Budapest_University_of_Technology_and_Economics"
COUNTRY="Hungary"

TRAIN_TEST_SPLIT=private_test_hard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/openscene_meta_datas
CHECKPOINT=/root/workdir/NAVSIM/exp/coa_dinovits8_addition/2025.05.06.22.11.08/camera_only/799a9cch/checkpoints/epoch35.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle_challenge.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=camera_only_agent \
experiment_name=coa_dinovits8_addition_submission \
agent.checkpoint_path=$CHECKPOINT \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
