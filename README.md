# TEXT_BERT_CNN
在 Google BERT Fine-tuning基础上，利用cnn进行中文文本的分类;<br>
<br>
没有使用tf.estimator API接口的方式实现，主要我不太熟悉，也不习惯这个API，还是按原先的[text_cnn](https://github.com/cjymz886/text-cnn)实现方式来的;<br>
<br>
训练结果：在验证集上准确率是96.4%左右，训练集是100%；，这个结果单独利用cnn也是可以达到的。这篇blog不是来显示效果如何，主要想展示下如何在bert的模型下Fine-tuning，觉得以后这种方式会成为主流。<br>

1 环境
=
python3<br>
tensorflow 1.9.0以上

2 数据
=
还是以前的数据集，涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']；<br>
下载链接:[https://pan.baidu.com/s/11AuC5g47rnsancf6nfKdiQ] 密码:1vdg<br>

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

3 运行
=
python text_run.py train<br>
<br>
python text_run.py test<br>

4 结论
=
我个人感觉在bert基础上对text_cnn提升并不大，不过这个数据集我优化的最好结果在验证集上也只是97%左右，怀疑数据集中可能有些文本的类别不是特别明显，或是属于多个类别也是合理的<br>
<br>
bert在中文上目前只是支持字符级别的，而且文本长度最大为128，这个长度相对于单独卷积就处于劣势<br>
<br>
bert会导致运行效率降低很多，毕竟模型的参数量摆在那里，实际应用要一定的硬件支持<br>

5 参考
=
1. [google-research/bert](https://arxiv.org/abs/1408.5882)
2. [brightmart/bert_language_understanding](https://github.com/brightmart/bert_language_understanding)


![image](https://github.com/cjymz886/sentence-similarity/blob/master/images/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E7%AE%97%E6%B3%95%E4%B8%8E%E5%AE%9E%E8%B7%B5.png)<br>
