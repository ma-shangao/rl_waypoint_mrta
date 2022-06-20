#$ -l mem=128G
#$ -l h_rt=12:0:0
#$ -N train_100_3
#$ -wd /home/ucemaxx/Scratch/GNNClustering

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate GNNClustering

python /home/ucemaxx/Scratch/GNNClustering/main.py --train -N 100 -M 64 --model_type mlp --lamb 0
