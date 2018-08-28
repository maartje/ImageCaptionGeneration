#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=09:00:00
#PBS -qgpu

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/preprocess "$TMPDIR"

mkdir -p "$TMPDIR"/train/show_attend_tell_sgd_1
cp "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1/* "$TMPDIR"/train/show_attend_tell_sgd_1

mkdir -p "$TMPDIR"/train/show_attend_tell_sgd_1_alpha_01
cp "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1_alpha_01/* "$TMPDIR"/train/show_attend_tell_sgd_1_alpha_01

#Run program
CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k_resume.json --model_dir show_attend_tell_sgd_1 --learning_rate 0.25 --optimizer SGD  --model show_attend_tell --alpha_c 1.0 --fname_resume model.21.pt &

CUDA_VISIBLE_DEVICES=1 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k_resume.json --model_dir show_attend_tell_sgd_1_alpha_01 --learning_rate 0.25 --optimizer SGD  --model show_attend_tell --alpha_c 0.1 --fname_resume model.21.pt &

wait

#Copy output data to persistent disk 
mkdir -p "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1_resume
cp "$TMPDIR"/train/show_attend_tell_sgd_1/* "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1_resume

mkdir -p "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1_alpha_01_resume
cp "$TMPDIR"/train/show_attend_tell_sgd_1_alpha_01/* "$HOME"/ImageCaptionGeneration/flickr30k/train/show_attend_tell_sgd_1_alpha_01_resume


