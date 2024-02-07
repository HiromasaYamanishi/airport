CUDA_VISIBLE_DEVICES=4 python posneg_emb.py --ind 0 --device 4 --model luke --suffix incontext_kagawa &
CUDA_VISIBLE_DEVICES=5 python posneg_emb.py --ind 1 --device 5 --model luke --suffix incontext_kagawa &
CUDA_VISIBLE_DEVICES=6 python posneg_emb.py --ind 2 --device 6 --model luke --suffix incontext_kagawa &
wait