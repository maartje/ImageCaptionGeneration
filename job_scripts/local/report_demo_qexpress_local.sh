#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/data   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/train   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/predict "$TMPDIR"

#Run program
cd "$HOME"/ImageCaptionGeneration
python3 -W ignore report.py --config configs/config_lisa.json --model_dir show_tell_adam_min3

#Copy output data to persistent disk 
cp -r "$TMPDIR"/report "$HOME"/ImageCaptionGeneration/flickr_demo


