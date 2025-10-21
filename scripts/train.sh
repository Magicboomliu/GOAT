
TRAINSF_DDP(){
# Navigate to project root
cd "$(dirname "$0")/.."

pretrain_name=OGMNet_revise14
mkdir -p logs
loss=configs/loss_config_disp.json
outf_model=models_saved/$pretrain_name
logf=logs/$pretrain_name
datapath=/media/zliu/datagrid1/liu/sceneflow
datathread=4
lr=1e-3
devices=0
dataset=sceneflow
trainlist=data/filenames/SceneFlow/SceneFlow_With_Occ.list
vallist=data/filenames/SceneFlow/FlyingThings3D_Test_With_Occ.list
startR=0
startE=0
batchSize=1
testbatch=4
maxdisp=-1
save_logdir=experiments_logdir/$pretrain_name
model=$pretrain_name
pretrain=none
initial_pretrain=none

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 scripts/train.py --cuda --loss $loss --lr $lr \
               --outf $outf_model --logFile $logf \
               --devices $devices --batch_size $batchSize \
               --dataset $dataset --trainlist $trainlist --vallist $vallist \
               --startRound $startR --startEpoch $startE \
               --model $model \
               --maxdisp $maxdisp \
               --datapath $datapath \
               --manualSeed 1024 --test_batch $testbatch \
               --save_logdir $save_logdir \
               --pretrain $pretrain \
               --initial_pretrain $initial_pretrain \
               --datathread $datathread
}


TRAINSF_DDP