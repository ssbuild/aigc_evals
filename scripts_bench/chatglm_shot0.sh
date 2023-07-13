# 修改 constant_map.py 权重配置
python ../eval.py  --device=3 --data_type="ceval" --model_name="chatglm-6b"
python ../eval.py  --device=3 --data_type="cmmlu" --model_name="chatglm-6b"