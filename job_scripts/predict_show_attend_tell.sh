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
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/train "$TMPDIR"

#Run program
CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_tell_sgd_1_fixed &

CUDA_VISIBLE_DEVICES=2 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min4 &

CUDA_VISIBLE_DEVICES=3 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min5 &

wait

CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_sgd_1 &

CUDA_VISIBLE_DEVICES=1 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_sgd_1_alpha_01 &

CUDA_VISIBLE_DEVICES=2 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min3 &

CUDA_VISIBLE_DEVICES=3 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min2 &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/predict "$HOME"/ImageCaptionGeneration/flickr30k


