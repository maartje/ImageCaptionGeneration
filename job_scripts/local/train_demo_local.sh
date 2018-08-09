#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_sgd_1 --learning_rate 1.0 --optimizer SGD &

python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_adam_min5 --learning_rate 0.00001 &

python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_adam_min4 --learning_rate 0.0001 &

python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_adam_min3 --learning_rate 0.001 &

vmstat 60 30

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr_demo


