#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=0:59:00
#PBS -qexpress_gpu
# -qgpu

#Loading modules
# TODO
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Copy input data to scratch and create output directory
mkdir "$TMPDIR"/output
cp -r "$HOME"/ImageCaptionGeneration/flickr30k "$TMPDIR"
echo "$TMPDIR"/flickr30k
echo $TMPDIR/flickr30k
echo "   do ls on TMPDIR/flickr30k"
ls "$TMPDIR"/flickr30k
ls $TMPDIR/flickr30k

#Run program
cd ImageCaptionGeneration
#python -W ignore -m unittest discover -v
python3 -W ignore train.py --config config_lisa.json

#Copy output data from scratch to home
cp -r "$TMPDIR"/output "$HOME"/ImageCaptionGeneration
