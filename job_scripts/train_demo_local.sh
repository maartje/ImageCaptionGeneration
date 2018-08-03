#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_1 --learning_rate 1.0 &

python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_05 --learning_rate 0.5 &

python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_maartje.json --model_dir show_tell_sgd_2 --learning_rate 2.0 &

wait

#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr_demo


