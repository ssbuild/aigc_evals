# 修改 constant_map.py 权重配置
python ../eval.py  --device=1 --data_type="ceval" --k=5 --few_shot --model_name="baichuan-7B"
python ../eval.py  --device=1 --data_type="cmmlu" --k=5 --few_shot --model_name="baichuan-7B"
