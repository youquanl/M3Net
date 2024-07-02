#!/bin/sh

TASK=train
IMAGE=docker://pointcept/pointcept:pytorch1.11.0-cuda11.3-cudnn8-devel
PYTHON=/opt/conda/bin/python
DATASET=scannet
CONFIG="None"
EXP_NAME=debug

NUM_GPU=4
NUM_MACHINE=1
NUM_CPU=None
NUM_CPU_PER_GPU=16

WEIGHT="None"
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
#tmux send-keys -t "${TASK}-${DATASET}-${EXP_NAME}" "export APPTAINER_CACHEDIR=/var/tmp"
if [ "${TASK}" == 'train' ]
then
  tmux send-keys -t "${TASK}-${DATASET}-${EXP_NAME}" "
  srun --preempt -u -p Ai4sci_3D --job-name=${TASK}-${DATASET}-${EXP_NAME} --gres=gpu:$NUM_GPU --nodes=$NUM_MACHINE --ntasks-per-node=1 --cpus-per-task=$NUM_CPU --kill-on-bad-exit \
  apptainer exec --nv --bind /mnt:/mnt ${IMAGE} \
  sh $SCRIPTS_DIR/train.sh -p $PYTHON -g $NUM_GPU -d $DATASET -c $CONFIG -n $EXP_NAME -w $WEIGHT -r $RESUME
  " Enter
fi

if [ "${TASK}" == 'test' ]
then
  tmux send-keys -t "${TASK}-${DATASET}-${EXP_NAME}" "
  srun --preempt -u -p Ai4sci_3D --job-name=${TASK}-${DATASET}-${EXP_NAME} --gres=gpu:$NUM_GPU --nodes=$NUM_MACHINE --ntasks-per-node=1 --cpus-per-task=$NUM_CPU --kill-on-bad-exit \
  apptainer exec --nv --bind /mnt:/mnt ${IMAGE} \
  sh $SCRIPTS_DIR/test.sh -p $PYTHON -g $NUM_GPU -d $DATASET -n $EXP_NAME -w $WEIGHT
  " Enter
fi
