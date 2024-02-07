#python clustering_and_visualize.py --suffix incontext_kagawa_luke --div 3
CUDA_VISIBLE_DEVICES=2 python summarize_topic.py --suffix incontext_kagawa_luke --extra_suffix male
CUDA_VISIBLE_DEVICES=2 python summarize_topic.py --suffix incontext_kagawa_luke --extra_suffix female