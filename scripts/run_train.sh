#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================




config=$2

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ./configs ./train_parallel$i
    cp -r ./data ./train_parallel$i
    cp -r ./scripts ./train_parallel$i
    cp -r ./trainers ./train_parallel$i
    cp -r ./utils ./train_parallel$i
    cp ./train.py ./train_parallel$i
    cp ./args.py ./train_parallel$i
    cp ./__init__.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID, $config"
    cd ./train_parallel$i ||exit
    env > env.log
    python train.py --device-id $i --config $config > train$i.log 2>&1 &
    cd ..
done
