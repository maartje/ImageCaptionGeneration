#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=2:00:00
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
CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_sgd_1_resume_decay &

CUDA_VISIBLE_DEVICES=1 python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config.json --model_dir show_tell_sgd_1_resume &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/predict "$HOME"/ImageCaptionGeneration/flickr30k


