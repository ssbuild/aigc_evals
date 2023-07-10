# -*- coding: utf-8 -*-
# @Time:  19:26
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments, DataHelper
from transformers import HfArgumentParser
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer
from aigc_zoo.utils.llm_generate import Generate
class NN_DataHelper(DataHelper):pass


train_info_models = {
    'bloom-560m': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    },
    'bloom-1b7': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    },
    'opt-350m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
        'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',
    },

    'llama-7b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    },


}

train_info_args = {
    'data_backend': 'parquet',
    # 预训练模型路径
    **train_info_models['bloom-560m'],
    'use_fast_tokenizer': False,
    'do_lower_case': None,
}


class Engine_API:
    def init(self):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()

        pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=config.torch_dtype, )
        model = pl_model.get_llm_model()

        model.eval().half().cuda()

        self.model = model
        self.tokenizer = tokenizer

    def infer(self,input,**kwargs):
        default_kwargs = dict(
            max_length=512,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=False, top_p=0.7, temperature=0.95,
        )
        kwargs.update(default_kwargs)
        response = Generate.generate(self.model,
                                     tokenizer=self.tokenizer,
                                     query=input,**kwargs)
        return response

if __name__ == '__main__':
    api_client = Engine_API()
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]
    for input in text_list:
        response = api_client.infer(input)
        print('input', input)
        print('output', response)