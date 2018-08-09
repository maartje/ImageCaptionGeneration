#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=8:00:00
#PBS -qgpu

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/preprocess "$TMPDIR"

#Run program
CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_sgd_1 --learning_rate 1.0 --optimizer SGD &

CUDA_VISIBLE_DEVICES=1 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_adam_5min5 --learning_rate 0.00005 &

CUDA_VISIBLE_DEVICES=2 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_adam_5min4 --learning_rate 0.0005 &

CUDA_VISIBLE_DEVICES=3 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_adam_5min3 --learning_rate 0.005 &

sleep 15
for i in {1..6};
do
    sleep 10
    vmstat
    nvidia-smi
done

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr30k


