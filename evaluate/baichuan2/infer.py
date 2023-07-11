# -*- coding: utf-8 -*-
# @Time:  18:49
# @Author: tk
# @File：evaluate
import torch
from deep_training.data_helper import ModelArguments, DataArguments,DataHelper
from transformers import HfArgumentParser, BitsAndBytesConfig
from aigc_zoo.model_zoo.baichuan2.llm_model import MyTransformer,BaichuanConfig,BaichuanTokenizer
from aigc_zoo.utils.llm_generate import Generate
class NN_DataHelper(DataHelper):pass

train_info_args = {
    'data_backend': 'parquet',
    # 预训练模型路径
    'model_type': 'baichuan',
    'model_name_or_path': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat',
    'config_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat/config.json',
    'tokenizer_name': '/data/nlp/pre_models/torch/baichuan/Baichuan-13B-Chat',
    'use_fast_tokenizer': False,
    'do_lower_case': None,
}


class Engine_API:
    def init(self):
        parser = HfArgumentParser((ModelArguments,))
        (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

        dataHelper = NN_DataHelper(model_args)
        tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                                       tokenizer_class_name=BaichuanTokenizer)
        config.pad_token_id = config.eos_token_id

        pl_model = MyTransformer(config=config, model_args=model_args,
                                 torch_dtype=torch.float16, )

        model = pl_model.get_llm_model()
        model = model.eval()
        model.requires_grad_(False)

        model = model.half().cuda()

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
                                     query=input,**default_kwargs)
        return response


if __name__ == '__main__':
    api_client = Engine_API()
    api_client.init()
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 "登鹳雀楼->王之涣\n夜雨寄北->",
                 "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
                 ]
    for input in text_list:
        response = api_client.infer(input)
        print('input', input)
        print('output', response)