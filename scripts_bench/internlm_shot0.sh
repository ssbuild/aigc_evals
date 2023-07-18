# 修改 constant_map.py 权重配置
python ../eval.py  --device=2 --data_type="ceval" --few_shot --model_name="internlm-chat-7b"
python ../eval.py  --device=2 --data_type="cmmlu" --few_shot --model_name="internlm-chat-7b"