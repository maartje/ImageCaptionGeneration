#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=0:00:30
# -qexpress_gpu
# -qgpu

#Loading modules
# TODO
# source activate summer_project
source activate summer_project
#module load python/3.7.0
module load eb
module load CUDA
#module load cudnn

#Copy input data to scratch and create output directory
#mkdir "$TMPDIR"/flickr30k
#mkdir "$TMPDIR"/output
#cp -r $HOME/flickr30k "$TMPDIR"/flickr30k

#Run program
#python3 -W ignore train.py --config config_lisa.json
#which nvcc
#echo 'mj'
#nvcc --version
echo 'hello world!' 

python3 ImageCaptionGeneration/bar.py
#echo 'hello from bash!' 

#Copy output data from scratch to home
#cp -r "$TMPDIR"/output $HOME
