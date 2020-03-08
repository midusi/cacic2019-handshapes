declare -A rwth_config

# rwth_config["data.train_way"]=
# rwth_config["data.train_support"]=
# rwth_config["data.train_query"]=
# rwth_config["data.test_way"]=
# rwth_config["data.test_support"]=
# rwth_config["data.test_query"]=
# rwth_config["data.episodes"]=
# rwth_config["data.cuda"]=
# rwth_config["data.gpu"]=

# rwth_config["model.base"]="DenseNet"

rwth_config["data.rotation_range"]="10"
rwth_config["data.width_shift_range"]="0.10"
rwth_config["data.height_shift_range"]="0.10"
rwth_config["data.horizontal_flip"]="True"

rwth_config["data.batch_size"]="32"
# rwth_config["data.train_size"]=
# rwth_config["data.test_size"]=
# rwth_config["data.n_train_per_class"]=
# rwth_config["data.n_test_per_class"]=

rwth_config["train.epochs"]="50"
rwth_config["train.patience"]="15"
rwth_config["train.lr"]="0.001"

# rwth_config["model.x_dim"]=
# rwth_config["model.z_dim"]=
# rwth_config["model.type"]=
