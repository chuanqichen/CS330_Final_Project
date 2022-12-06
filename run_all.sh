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



CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 12 --num_support 1 --num_query 1  --log_dir ./temp & 
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

python maml.py --test --num_way 2 --num_support 1 --num_query 1 --log_dir ./logs/maml/crop.way:2.support:1.query:1.lr:0.001.batch_size:16/ --checkpoint_step 4900 --data_folder ./data/meta_learning_part_5.csv