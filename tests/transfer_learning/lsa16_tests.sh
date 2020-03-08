declare -A lsa16_config

lsa16_config["data.batch_size"]="32"

lsa16_config["data.rotation_range"]="10"
lsa16_config["data.width_shift_range"]="0.10"
lsa16_config["data.height_shift_range"]="0.10"
lsa16_config["data.horizontal_flip"]="True"

lsa16_config["data.train_size"]="0.8"
lsa16_config["data.test_size"]="0.2"
lsa16_config["data.n_train_per_class"]="0"
lsa16_config["data.n_test_per_class"]="0"

lsa16_config["data.weight_classes"]="True"
lsa16_config["model.name"]="DenseNet"

lsa16_config["train.epochs"]="150"
lsa16_config["train.patience"]="25"
lsa16_config["train.lr"]="0.001"
