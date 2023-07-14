# 修改 constant_map.py 权重配置
python ../eval.py  --device=2 --data_type="ceval" --k=5 --few_shot --model_name="chatglm2-6b-int4"
python ../eval.py  --device=2 --data_type="cmmlu" --k=5 --few_shot --model_name="chatglm2-6b-int4"