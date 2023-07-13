# 修改 constant_map.py 权重配置
python ../eval.py  --device=0 --data_type="ceval" --model_name="Baichuan-13B-Chat"
python ../eval.py  --device=0 --data_type="cmmlu" --model_name="Baichuan-13B-Chat"