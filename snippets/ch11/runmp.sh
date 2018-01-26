#!/bin/bash 

	
for run in {1..10}
do
  python mp_train.py >> results_t.txt
done
