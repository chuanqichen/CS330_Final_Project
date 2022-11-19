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