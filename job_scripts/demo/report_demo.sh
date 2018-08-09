#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=0:02:00

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/data   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/train   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/predict "$TMPDIR"

#Run program
cd "$HOME"/ImageCaptionGeneration
python3 -W ignore report.py --config configs/config_lisa.json --model_dir show_tell_sgd_1 &

python3 -W ignore report.py --config configs/config_lisa.json --model_dir show_tell_adam_min5 &

python3 -W ignore report.py --config configs/config_lisa.json --model_dir show_tell_adam_min4 &

python3 -W ignore report.py --config configs/config_lisa.json --model_dir show_tell_adam_min3 &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/report "$HOME"/ImageCaptionGeneration/flickr_demo


