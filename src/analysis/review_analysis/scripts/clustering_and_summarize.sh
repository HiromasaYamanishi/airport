#python clustering_and_visualize.py --suffix incontext_kagawa_luke --extra_suffix all_re --div 3 --df_path '../data/kagawa_review.csv'
GPU=1

#CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix incontext_kagawa_luke --extra_suffix all_re --summarize general --df_path '../data/kagawa_review.csv'
#CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix incontext_kagawa_luke --extra_suffix all_re --summarize simple --df_path '../data/kagawa_review.csv'
#CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix incontext_kagawa_luke --extra_suffix all_re --summarize detail --df_path '../data/kagawa_review.csv'

# python clustering_and_visualize.py --suffix hotel --extra_suffix re --div 3 --df_path ../data/hotel/review/review_kagawa.csv
# GPU=1

# CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix hotel --extra_suffix re --summarize general --df_path ../data/hotel/review/review_kagawa.csv
# CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix hotel --extra_suffix re --summarize simple --df_path ../data/hotel/review/review_kagawa.csv
# CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix hotel --extra_suffix re --summarize detail --df_path ../data/hotel/review/review_kagawa.csv

python clustering_and_visualize.py --suffix food --extra_suffix re --div 3 --df_path ../data/review_food_all_period_.csv
GPU=4

CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix food --extra_suffix re --summarize general --df_path ../data/review_food_all_period_.csv
CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix food --extra_suffix re --summarize simple --df_path ../data/review_food_all_period_.csv
CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix food --extra_suffix re --summarize detail --df_path ../data/review_food_all_period_.csv