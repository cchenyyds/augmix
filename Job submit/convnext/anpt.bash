#!/usr/bin/bash
#SBATCH -J convnext_tiny_apt
#SBATCH -o convnext_tiny_pt-b.o
#SBATCH --time=5:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH -e convnext_tiny_b.o


module load GpuModules

module unload tensorflow2-py37-cuda11.2-gcc8/2.5.2

source ~/miniconda3/etc/profile.d/conda.sh

conda activate /home/g060677/miniconda3/envs/hh



cd /work/ws-tmp/g060677-Cc/augmix-master1

python cifar_testcon1.py -m convnext_tiny  -opt AdamW -sch Cos -s ./convnext_tiny_anpt

#python cifar_testcon1.py -m convnext_tiny -opt SGD -sch Lam -s ./convnext_tiny_bnpt

