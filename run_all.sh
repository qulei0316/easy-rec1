#!/bin/bash
# 一键运行全流程脚本
# 说明：依次执行数据生成、特征工程、召回、排序、重排与融合
set -e

# echo "1. 生成数据..."
# python3 data/generate_data.py

# echo "2. 特征工程..."
# python3 features/build_features.py

echo "3. 召回..."
python3 recall/recall_main.py

echo "4. 粗排..."
python3 prerank/prerank_main.py

echo "5. 精排..."
python3 rank/rank_main.py

echo "6. 重排..."
python3 rerank/rerank_main.py

echo "7. 策略融合..."
python3 blend/blend_main.py

echo "流程完成！"
