export CUDA_VISIBLE_DEVICES='0'

MODE='train'
MODEL_LIST='STID,GWN,STWave'
load_train_paths='STID_PEMS08.pth,GWN_PEMS08.pth,STWave_PEMS08.pth'
dataset_use='PEMS08'

python ./Run.py -mode $MODE -model_list $MODEL_LIST \
        -load_train_paths $load_train_paths \
        -dataset_use $dataset_use \
        -batch_size 32 \
