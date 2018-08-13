#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=04:00:00
#PBS -qgpu

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/preprocess "$TMPDIR"
mkdir -p "$TMPDIR"/train/show_tell_sgd_1
cp "$HOME"/ImageCaptionGeneration/flickr30k/train/show_tell_sgd_1/* "$TMPDIR"/train/show_tell_sgd_1

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_resume.json --model_dir show_tell_sgd_1 --learning_rate 1.0 --optimizer SGD


#Copy output data to persistent disk 
mkdir -p "$HOME"/ImageCaptionGeneration/flickr30k/train/show_tell_sgd_1_resume
cp "$TMPDIR"/train/show_tell_sgd_1/* "$HOME"/ImageCaptionGeneration/flickr30k/train/show_tell_sgd_1_resume


