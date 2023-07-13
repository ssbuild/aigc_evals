# 修改 constant_map.py 权重配置
python ../eval.py  --device=0 --data_type="ceval" --few_shot --model_name="Baichuan-13B-Chat"
python ../eval.py  --device=0 --data_type="cmmlu" --few_shot --model_name="Baichuan-13B-Chat"