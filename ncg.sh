#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=0:05:00

#Loading modules
# TODO
source activate summer_project

#Copy input data to scratch and create output directory
mkdir "$TMPDIR"/flickr30k
mkdir "$TMPDIR"/output
cp -r $HOME/flickr30k "$TMPDIR"/flickr30k

#Run program
python3 -W ignore train.py --config config_lisa.json

#Copy output data from scratch to home
cp -r "$TMPDIR"/output $HOME
