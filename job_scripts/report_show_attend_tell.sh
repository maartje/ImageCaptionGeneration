#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=8:00:00

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/data   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/train   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/predict "$TMPDIR"

#Run program
cd "$HOME"/ImageCaptionGeneration
python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_tell_sgd_1_fixed &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min4_01 &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min4 &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min5 &

wait

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_sgd_1 &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_sgd_1_alpha_01 &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min3 &

python3 -W ignore report.py --config configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min2 &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/report "$HOME"/ImageCaptionGeneration/flickr30k


