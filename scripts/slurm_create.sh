#!/bin/sh

TASK=train
PYTHON=python
DATASET=scannet
CONFIG="None"
EXP_NAME=debug
PARTITION=None
NUM_GPU=8
NUM_MACHINE=1
NUM_CPU=None
NUM_CPU_PER_GPU=10

WEIGHT=model_best0
RESUME=false

while getopts "t:p:d:c:n:g:m:u:w:r" opt; do
  case $opt in
    t)
      TASK=$OPTARG
      ;;
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    u)
      NUM_CPU=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

SCRIPTS_DIR=$(cd $(dirname "$0");pwd) || exit

if [ "${NUM_CPU}" == 'None' ]
then
  NUM_CPU=`expr $NUM_CPU_PER_GPU \* $NUM_GPU`
fi

tmux new-session -d -s "${TASK}-${DATASET}-${EXP_NAME}"

if [ "${TASK}" == 'train' ]
then
  srun --preempt -u -p $PARTITION --job-name=train-scannet-semseg-pt-v7m1-0-base --gres=gpu:$NUM_GPU --nodes=$NUM_MACHINE --ntasks-per-node=1 --cpus-per-task=$NUM_CPU  --kill-on-bad-exit \
  sh $SCRIPTS_DIR/train.sh -p $PYTHON -g $NUM_GPU -d $DATASET -c $CONFIG -n $EXP_NAME -w $WEIGHT -r $RESUME


fi

if [ "${TASK}" == 'test' ]
then
  srun --preempt -u -p $PARTITION --job-name=${TASK}-${DATASET}-${EXP_NAME} --gres=gpu:$NUM_GPU  --nodes=$NUM_MACHINE  --ntasks-per-node=1 --quotatype=spot --cpus-per-task=$NUM_CPU --kill-on-bad-exit \
  sh $SCRIPTS_DIR/test.sh -p $PYTHON -g $NUM_GPU -d $DATASET -n $EXP_NAME -w $WEIGHT
fi
