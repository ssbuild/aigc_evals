# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/7 16:24
from rouge import Rouge

hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

rouger = Rouge()
scores = rouger.get_scores([hypothesis,hypothesis], [reference,reference],avg=True)

print(len(scores))
print(scores)