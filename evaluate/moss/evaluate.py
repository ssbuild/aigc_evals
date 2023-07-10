# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer,MossConfig,MossTokenizer
from aigc_zoo.utils.moss_generate import Generate
class NN_DataHelper(DataHelper):pass

train_info_args = {
    'data_backend': 'parquet',
    # 预训练模型路径
    'model_type': 'moss',
    'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4',
    'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4',
    'use_fast_tokenizer': False,
    'do_lower_case': None,
}

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments, ))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: MossTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer, config_class_name=MossConfig,
                                                                  config_kwargs={"torch_dtype": "float16"})

    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,)
    model = pl_model.get_llm_model()
    model.eval().half().cuda()

    gen_core = Generate(model,tokenizer)

    query =  "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:"
    response = gen_core.chat(query, max_length=2048,
                          # do_sample=False, top_p=0.7, temperature=0.95,
                          )
    print(query,' 返回: ',response)
