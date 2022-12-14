python MANN.py --num_shot 1 --num_classes 2  --random_seed 100 --log_dir ./temp & 
python MANN.py --num_shot 1 --num_classes 3  --random_seed 100 --log_dir ./temp & 
python MANN.py --num_shot 1 --num_classes 4  --random_seed 100 --log_dir ./temp & 
CUDA_VISIBLE_DEVICES=1 python MANN.py --num_shot 2 --num_classes 2  --random_seed 100 --log_dir ./temp & 
CUDA_VISIBLE_DEVICES=1 python MANN.py --num_shot 2 --num_classes 3  --random_seed 100 --log_dir ./temp & 
CUDA_VISIBLE_DEVICES=1 python MANN.py --num_shot 2 --num_classes 4  --random_seed 100 --log_dir ./temp & 

python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-2 &
python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-4 &
python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-5 &

python MANN.py --num_shot 1 --num_classes 3  --hidden_dim 256 & 
python MANN.py --num_shot 1 --num_classes 3  --hidden_dim 8 & 

python maml.py --num_way 2 --num_support 1 --num_query 1 &

CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 10 --num_support 1 --num_query 1  --log_dir ./temp & 
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 12 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 11 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 10 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 9 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 8 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 7 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 6 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 5 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 4 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 3 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 2 --num_support 1 --num_query 1 & 

CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 12 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 11 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 10 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 9 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 8 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 7 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 6 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 5 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 4 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 3 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 2 --num_support 1 --num_query 1  --log_dir ./logs_lstm_conv4 --data_folder ./data/v1.csv &

CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 4 --num_support 1 --num_query 1  --log_dir ./logs_new_temp  --data_folder ./data/v1.csv & 


CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 4 --num_support 1 --num_query 1  --log_dir ./logs_new_temp  --data_folder ./data/v1.csv & 

CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 12 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 11 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 10 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 9 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 8 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 7 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 6 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 5 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 4 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 3 --num_support 1 --num_query 1 & 
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 2 --num_support 1 --num_query 1 & 


python protonet.py --test --num_way 2 --num_support 1 --num_query 1 --log_dir ./logs/protonet/crop.way:2.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_4.csv

python protonet.py --test --num_way 2 --num_support 1 --num_query 1 --log_dir ./logs/protonet/crop.way:2.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv

CUDA_VISIBLE_DEVICES=1 python maml.py --test --num_way 2 --num_support 1 --num_query 1 --log_dir ./logs/maml/crop.way:5.support:2.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv


python protonet.py --test --num_way 2 --num_support 1 --num_query 2 --log_dir ./logs/protonet/crop.way:2.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.999, 95% confidence interval 0.001
python maml.py --test --num_way 2 --num_support 1 --num_query 2 --log_dir ./logs/maml/crop.way:2.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.833, 95% confidence interval 0.001

python protonet.py --test --num_way 3 --num_support 1 --num_query 2 --log_dir ./logs/protonet/crop.way:3.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.679, 95% confidence interval 0.004
python maml.py --test --num_way 3 --num_support 1 --num_query 2 --log_dir ./logs/maml/crop.way:3.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.827, 95% confidence interval 0.004

python protonet.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs/protonet/crop.way:3.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.706, 95% confidence interval 0.002
python maml.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs/maml/crop.way:3.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.799, 95% confidence interval 0.001



python protonet.py --test --num_way 3 --num_support 1 --num_query 15 --log_dir ./logs/protonet/crop.way:3.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.400, 95% confidence interval 0.000
python maml.py --test --num_way 3 --num_support 1 --num_query 15 --log_dir ./logs/maml/crop.way:3.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.400, 95% confidence interval 0.001

crop.way:4.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16



test
python protonet.py --test --num_way 2 --num_support 1 --num_query 5 --log_dir ./logs/protonet/crop.way:12.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 1.000, 95% confidence interval 0.000

python maml.py --test --num_way 2 --num_support 1 --num_query 5 --log_dir ./logs/maml/crop.way:6.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.698, 95% confidence interval 0.002


python protonet.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs/protonet/crop.way:12.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.600, 95% confidence interval 0.000

python maml.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs/maml/crop.way:6.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.465, 95% confidence interval 0.002

python protonet.py --test --num_way 4 --num_support 1 --num_query 5 --log_dir ./logs/protonet/crop.way:12.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.340, 95% confidence interval 0.002

python maml.py --test --num_way 4 --num_support 1 --num_query 5 --log_dir ./logs/maml/crop.way:6.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.354, 95% confidence interval 0.001

python protonet.py --test --num_way 5 --num_support 1 --num_query 5 --log_dir ./logs/protonet/crop.way:12.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.240, 95% confidence interval 0.000

python maml.py --test --num_way 5 --num_support 1 --num_query 5 --log_dir ./logs/maml/crop.way:6.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16/ --checkpoint_step 10000 --data_folder ./data/meta_learning_part_5.csv
Accuracy over 600 test tasks: mean 0.279, 95% confidence interval 0.001

python maml.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs_lstm_conv4/maml/crop.way:12.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16  --checkpoint_step 4000 --data_folder ./data/meta_learning_part_5.csv


num_way = 6, 0.2 
Accuracy over 600 test tasks: mean 0.200, 95% confidence interval 0.000
Accuracy over 600 test tasks: mean 0.204, 95% confidence interval 0.001


find ./logs_lstm/ -name 'events.out.tfevents.*' -exec cp -prv '{}' './experiments/' ';'
shopt -s globstar
cp --reflink=auto --parents **/events.out.tfevents.* ../experiments/




CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 4 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 3 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=0 python maml.py --num_way 2 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &

CUDA_VISIBLE_DEVICES=0 python protonet.py --num_way 4 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 3 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &
CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 2 --num_support 1 --num_query 1  --log_dir ./logs_new2_lstm_conv4 --data_folder ./data/v1.csv &


python protonet.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs_new_lstm_conv4/protonet/crop.way:3.support:1.query:1.lr:0.001.batch_size:16/  --checkpoint_step 900 --data_folder ./data/meta_learning_part_5.csv


python maml.py --test --num_way 3 --num_support 1 --num_query 5 --log_dir ./logs_new_lstm_conv4/maml/crop.way:3.support:1.query:1.inner_steps:1.inner_lr:0.4.learn_inner_lrs:False.outer_lr:0.001.batch_size:16  --checkpoint_step 900 --data_folder ./data/meta_learning_part_5.csv


