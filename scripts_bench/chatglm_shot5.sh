# 修改 constant_map.py 权重配置
python ../eval.py  --device=3 --data_type="ceval" --few_shot --model_name="chatglm-6b"
python ../eval.py  --device=3 --data_type="cmmlu" --few_shot --model_name="chatglm-6b"