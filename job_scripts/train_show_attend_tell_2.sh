#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=16:00:00
#PBS -qgpu

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr30k/preprocess "$TMPDIR"

#Run program
CUDA_VISIBLE_DEVICES=0 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_tell_sgd_1_fixed --learning_rate 1.0 --optimizer SGD  --model show_tell &

CUDA_VISIBLE_DEVICES=2 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min4_01 --learning_rate 0.01 --optimizer ADAM  --model show_attend_tell  --alpha_c 0.1 &

CUDA_VISIBLE_DEVICES=2 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min4 --learning_rate 0.01 --optimizer ADAM  --model show_attend_tell  --alpha_c 1.0 &

CUDA_VISIBLE_DEVICES=3 python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_flickr30k.json --model_dir show_attend_tell_adam_1min5 --learning_rate 0.001 --optimizer ADAM  --model show_attend_tell  --alpha_c 1.0 &

sleep 30
for i in {1..3};
do
    sleep 20
    vmstat
    nvidia-smi
done

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr30k


