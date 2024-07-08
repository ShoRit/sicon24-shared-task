# sicon24-shared-task
Shared Task for the SICON 2024 task

# Step 1: Setting up

# install dependencies

pip install -r requirements.txt

# extract data and 
mkdir ckpts
mkdir csv_files
tar -xzvf data.tar.gz


# Step 2: Run the baseline Flan-T5 model on one of the indomain datasets

cd code
python3 run_cls_model.py --mode ID --task persuasion --model_name flan-t5-base --do_train 1 --do_predict 0 

# Step 3: Use the save model checkpoint for evaluation (on the same task or across tasks)

cd code
python3 run_cls_model.py --mode TF --src_task persuasion --tgt_task negotiation --do_train 0 --do_predict 1

