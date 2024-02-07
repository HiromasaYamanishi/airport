#AVAILABLE_GPUS=(0 1 2)
AVAILABLE_GPUS=(0 1 2)
#suffix="food" #suffixに含めるべき情報: 用いたsentence_embeddignのモデル, クラスタリングの次元数, 
suffix="food"
df_path='../data/review_food_all_period_.csv'
#df_path='../data/hotel/review/review_all_period.csv'
GPU_NUM=${#AVAILABLE_GPUS[@]}
echo ${GPU_NUM}
index=0
for gpu in ${AVAILABLE_GPUS[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu} python analyze_review4.py --suffix ${suffix} --div ${GPU_NUM} --ind ${index} --df_path ${df_path} --data_type ${suffix} & #対象とするデータ, GPUの数
    ((index++))
done
wait
index=0
for gpu in ${AVAILABLE_GPUS[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu} python posneg_emb.py --model luke --suffix ${suffix} --ind ${index} --device ${gpu} & #対象とするデータ, GPUの数
    ((index++))
done
wait

python clustering_and_visualize.py --suffix ${suffix} --div ${GPU_NUM} --df_path ${df_path}

GPU=0
CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix ${suffix} --summarize general --df_path ${df_path}
CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix ${suffix} --summarize simple --df_path ${df_path}
CUDA_VISIBLE_DEVICES=${GPU} python summarize_topic.py --suffix ${suffix} --summarize detail --df_path ${df_path}