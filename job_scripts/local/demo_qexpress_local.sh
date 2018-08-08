#Copy input data to scratch 
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/data   "$TMPDIR"
cp -r "$HOME"/ImageCaptionGeneration/flickr_demo/preprocess "$TMPDIR"

#train
python3 -W ignore "$HOME"/ImageCaptionGeneration/train.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir demo_qexpress --learning_rate 0.001

#predict
python3 -W ignore "$HOME"/ImageCaptionGeneration/predict.py --config "$HOME"/ImageCaptionGeneration/configs/config_lisa.json --model_dir demo_qexpress

#report
cd "$HOME"/ImageCaptionGeneration
python3 -W ignore report.py --config configs/config_lisa.json --model_dir demo_qexpress


#Copy output data to persistent disk 
cp -r "$TMPDIR"/train "$HOME"/ImageCaptionGeneration/flickr_demo
cp -r "$TMPDIR"/predict "$HOME"/ImageCaptionGeneration/flickr_demo
cp -r "$TMPDIR"/report "$HOME"/ImageCaptionGeneration/flickr_demo


