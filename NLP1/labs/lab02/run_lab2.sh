#!/bin/bash

train_files=("wiki.train.raw" 
             "wiki.train_18359.raw" 
             "wiki.train_9179.raw" 
             "wiki.train_4589.raw" 
             "wiki.train_2294.raw" 
             "wiki.train_1147.raw" 
             "wiki.train_573.raw" 
             "wiki.train_286.raw" 
             "wiki.train_143.raw" 
             "wiki.train_71.raw" 
             "wiki.train_35.raw")

for train_file in "${train_files[@]}"
do
    python3 lab2.py "$train_file"
done

