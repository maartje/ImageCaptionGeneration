#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/train "$TMPDIR"

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir show_tell_adam_min3

#Copy output data to persistent disk 
cp -r "$TMPDIR"/predict "$HOME"/ImageCaptionGeneration/flickr_demo


