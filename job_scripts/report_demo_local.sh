#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/data   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/train   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/predict "$TMPDIR"

#Run program
cd "$HOME"/ImageCaptionGeneration
python3 -W ignore report.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_1 &

python3 -W ignore report.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_05 &

python3 -W ignore report.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_2 &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/report "$HOME"/ImageCaptionGeneration/flickr_demo


