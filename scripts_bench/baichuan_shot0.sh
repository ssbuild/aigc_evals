# 修改 constant_map.py 权重配置
python ../eval.py  --device=1 --data_type="ceval" --model_name="baichuan-7B"
python ../eval.py  --device=1 --data_type="cmmlu" --model_name="baichuan-7B"
