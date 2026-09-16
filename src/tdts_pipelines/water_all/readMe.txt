本目录完成的工作是分水岭算法在有去噪算法情况下的表现
1 识别模型 tdts_crm、tdts_mrm、tdts_nrm更换
   手动更换 batch_main_recog2.py  line64 DIGIT_MODEL_PATH对应的文件
2 单字符训练集构成的双字符和单字符测试集构成的双字符更换
  手动更换 batch_main_recog2.py  line70 train的逻辑值