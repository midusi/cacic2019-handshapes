declare -A lsa16_config

# lsa16_config["data.train_way"]=
# lsa16_config["data.train_support"]=
# lsa16_config["data.train_query"]=
# lsa16_config["data.test_way"]=
# lsa16_config["data.test_support"]=
# lsa16_config["data.test_query"]=
# lsa16_config["data.episodes"]=
# lsa16_config["data.cuda"]=
# lsa16_config["data.gpu"]=

# lsa16_config["model.base"]="DenseNet"

lsa16_config["data.rotation_range"]="10"
lsa16_config["data.width_shift_range"]="0.10"
lsa16_config["data.height_shift_range"]="0.10"
lsa16_config["data.horizontal_flip"]="True"

lsa16_config["data.batch_size"]="32"
# lsa16_config["data.train_size"]=
# lsa16_config["data.test_size"]=
# lsa16_config["data.n_train_per_class"]=
# lsa16_config["data.n_test_per_class"]=

lsa16_config["train.epochs"]="50"
lsa16_config["train.patience"]="15"
lsa16_config["train.lr"]="0.001"

# lsa16_config["model.x_dim"]=
# lsa16_config["model.z_dim"]=
# lsa16_config["model.type"]=
