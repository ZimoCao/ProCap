#!/usr/bin/env bash
set -e

export PYTHONHASHSEED=42

GPUS=""
MODEL_TYPE=""
DATASET=""
TASK=""
MASTER_PORT="29515"
SCENE_SEEN_FLAG=""

EXP_NAME=$

REFINEMENT=true
MASK=true
SCENE_QFORMER=true
PROJ_QFORMER=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seen_scene)
      SEEN_SCENE_FLAG="--seen_scene"
      SCENE_SEEN_FLAG="seen"
      shift
      ;;
    --unseen_scene)
      UNSEEN_SCENE_FLAG="--unseen_scene"
      SCENE_SEEN_FLAG="unseen"
      shift
      ;;
    --newsetting)
      UNSEEN_SCENE_FLAG="--newsetting"
      SCENE_SEEN_FLAG="newsetting"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --no_refinement) REFINEMENT=false; shift ;;
    --no_mask) MASK=false; shift ;;
    --no_scene_qformer) SCENE_QFORMER=false; shift ;;
    --no_proj_qformer) PROJ_QFORMER=false; shift ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$GPUS" ]]; then
  echo "Error: --gpus is required (e.g. --gpus 1,2,3)"
  exit 1
fi

if [[ -z "$MODEL_TYPE" ]]; then
  echo "Error: --model_type is required"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPUS

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC"
echo "master_port=$MASTER_PORT"
echo "model_type=$MODEL_TYPE"
echo "dataset=$DATASET"
echo "task=$TASK"
echo "$SEEN_SCENE_FLAG"
echo "$UNSEEN_SCENE_FLAG"


STATE_ARGS=""
if [ "$REFINEMENT" = false ]; then 
  STATE_ARGS="$STATE_ARGS --disable_refinement"
  DISABLE_STR="no_refinement"
fi
if [ "$MASK" = false ]; then 
  STATE_ARGS="$STATE_ARGS --disable_mask"
  DISABLE_STR="no_mask"
fi
if [ "$SCENE_QFORMER" = false ]; then 
  STATE_ARGS="$STATE_ARGS --disable_scene_qformer"
  DISABLE_STR="no_scene_qformer"
fi
if [ "$PROJ_QFORMER" = false ]; then 
  STATE_ARGS="$STATE_ARGS --disable_proj_qformer"
  DISABLE_STR="no_proj_qformer"
fi

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..
EXP_NAME=procap_${MODEL_TYPE}_${DISABLE_STR}_${DATASET}_${TASK}_${SCENE_SEEN_FLAG}_scene
TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/eval/${EXP_NAME}
mkdir -p $LOG_FOLDER
LOG_FILE="$LOG_FOLDER/${DATASET}_${TIME_START}.log"

torchrun \
  --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT \
  eval_procap.py \
  --model_type $MODEL_TYPE \
  --dataset $DATASET \
  --task $TASK \
  $SEEN_SCENE_FLAG \
  $UNSEEN_SCENE_FLAG \
  --ckpt_path results/train_rgbp_procap_${MODEL_TYPE}_${DISABLE_STR}/000.pt \
  --device cuda \
  --out_path results/train_rgbp_procap_${MODEL_TYPE}_${DISABLE_STR}/generated_captions/ \
  $STATE_ARGS \
|& tee -a  ${LOG_FILE}

