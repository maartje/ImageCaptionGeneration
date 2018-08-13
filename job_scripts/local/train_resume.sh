#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"
mkdir -p "$TMPDIR"/train/show_tell_sgd_1
cp "$HOME"/ImageCaptionGeneration/flickr_demo/train/show_tell_sgd_1/* "$TMPDIR"/train/show_tell_sgd_1

#Run program
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_resume.json --model_dir show_tell_sgd_1 --learning_rate 1.0 --optimizer SGD


#Copy output data to persistent disk 
mkdir -p "$HOME"/ImageCaptionGeneration/flickr_demo/train/show_tell_sgd_1_resume
cp "$TMPDIR"/train/show_tell_sgd_1/* "$HOME"/ImageCaptionGeneration/flickr_demo/train/show_tell_sgd_1_resume


