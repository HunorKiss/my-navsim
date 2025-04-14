TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=camera_only_agent \
experiment_name=training_camera_only_agent \
trainer.params.max_epochs=50 \
train_test_split=$TRAIN_TEST_SPLIT \
