# sicon24-shared-task
Shared Task for the SICON 2024 task

# Step 1: Setting up

## Install dependencies

pip install -r requirements.txt

## Extract Data and create directories
mkdir ckpts
mkdir csv_files
tar -xzvf data.tar.gz

# Step 2: Training

## Run the baseline Flan-T5 model on one of the indomain datasets

cd code
python3 run_cls_model.py --mode ID --task persuasion --model_name flan-t5-base --do_train 1 --do_predict 0 

# Step 3: Evaluation

## Source and Target Tasks are the same

cd code
python3 run_cls_model.py --mode ID --task persuasion --do_train 0 --do_predict 1


## Source and Target Tasks differ

cd code
python3 run_cls_model.py --mode TF --src_task persuasion --tgt_task negotiation --do_train 0 --do_predict 1 --fewshot 0

## Carry out the actual evaluation 

python3 eval_code.py 
