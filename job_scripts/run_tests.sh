#Set job requirements
#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -lwalltime=0:00:40
#PBS -qexpress_gpu

#Loading modules
source activate summer_project
module load eb
module load CUDA
module load cuDNN

#Run program
cd "$HOME"/ImageCaptionGeneration
python -W ignore -m unittest discover -v


