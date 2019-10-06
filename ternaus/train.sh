#!/usr/bin/env bash
set -e # abort if any command fails
#source activate py35_ternaus

python src/prepare_folds.py

# just work with one fold for now
for i in 0  #1 2 3 4
do
   #python src/train.py --size 1280x1920 --device-ids 0,1,2,4 --batch-size 4 --fold $i --workers 12 --lr 0.0001 --n-epochs 52
   # to just work with one Collab GPU
   python src/train.py --size 1280x1920 --device-ids 0 --batch-size 4 --fold $i --workers 12 --lr 0.0001 --n-epochs 52
done

#source deactivate

