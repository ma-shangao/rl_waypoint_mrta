#$ -l mem=64G
#$ -l h_rt=12:0:0
#$ -N train_200_3
#$ -wd /home/ucemaxx/Scratch/GNNClustering

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate GNNClustering

python /home/ucemaxx/Scratch/GNNClustering/main.py --train -N 200 -M 64 -k 3 --sample_num 2000000 --penalty_score 35.0
