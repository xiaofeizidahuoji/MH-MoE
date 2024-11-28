export CUDA_VISIBLE_DEVICES='0'

MODE='test'
MODEL_LIST='STID,GWN,STWave'
load_train_path=''  
dataset_use='PEMS04'
exp_id=
python ./Run.py -mode $MODE -model_list $MODEL_LIST \
        -load_train_path $load_train_path \
        -dataset_use $dataset_use \
        -exp_id $exp_id \
        -batch_size 32
