declare -A rwth_config

rwth_config["data.batch_size"]="32"

rwth_config["data.rotation_range"]="10"
rwth_config["data.width_shift_range"]="0.10"
rwth_config["data.height_shift_range"]="0.10"
rwth_config["data.horizontal_flip"]="True"

rwth_config["data.train_size"]="0.8"
rwth_config["data.test_size"]="0.2"
rwth_config["data.n_train_per_class"]="0"
rwth_config["data.n_test_per_class"]="0"

rwth_config["data.weight_classes"]="True"
rwth_config["model.name"]="DenseNet"

rwth_config["train.epochs"]="150"
rwth_config["train.patience"]="25"
rwth_config["train.lr"]="0.001"
