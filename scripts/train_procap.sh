#!/usr/bin/env bash
set -e

export PYTHONHASHSEED=42

GPUS=""
MODEL_TYPE=""
MASTER_PORT="29535"

REFINEMENT=true
MASK=true
SCENE_QFORMER=true
PROJ_QFORMER=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
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

export CUDA_VISIBLE_DEVICES=$GPUS

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC"
echo "model_type=$MODEL_TYPE"
echo "master_port=$MASTER_PORT"

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..
EXP_NAME=procap_${MODEL_TYPE}_${DISABLE_STR}
TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/eval/${EXP_NAME}
mkdir -p $LOG_FOLDER
LOG_FILE="$LOG_FOLDER/${TIME_START}.log"

torchrun \
  --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT \
  train_procap.py \
  --model_type $MODEL_TYPE \
  --out_dir results/train_rgbp_procap_${MODEL_TYPE}_${DISABLE_STR} \
  $STATE_ARGS \
|& tee -a  ${LOG_FILE}
