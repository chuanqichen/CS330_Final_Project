python MANN.py --num_shot 1 --num_classes 2  --random_seed 100 & 
python MANN.py --num_shot 1 --num_classes 3  --random_seed 100 & 
python MANN.py --num_shot 1 --num_classes 4  --random_seed 100 & 
python MANN.py --num_shot 2 --num_classes 2  --random_seed 100 & 
python MANN.py --num_shot 2 --num_classes 3  --random_seed 100 & 
python MANN.py --num_shot 2 --num_classes 4  --random_seed 100 & 

python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-2 &
python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-4 &
python MANN.py --num_shot 1 --num_classes 3  --learning_rate 1e-5 &

python MANN.py --num_shot 1 --num_classes 3  --hidden_dim 256 & 
python MANN.py --num_shot 1 --num_classes 3  --hidden_dim 8 & 

python maml.py --num_way 2 --num_support 1 --num_query 1 &

CUDA_VISIBLE_DEVICES=1 python maml.py --num_way 10 --num_support 1 --num_query 1 & 

CUDA_VISIBLE_DEVICES=1 python protonet.py --num_way 5 --num_support 1 --num_query 1 & 
