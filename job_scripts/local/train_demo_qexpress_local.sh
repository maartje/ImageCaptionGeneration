#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_adam_min3 --learning_rate 0.001 &

vmstat 15 30

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr_demo 



