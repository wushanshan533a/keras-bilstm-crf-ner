# 基于keras+Bilstm+crf的中文命名实体识别
这是一个基于keras+Bilstm+crf的中文命名实体识别练习，主要是理解命名实体识别的做法以及bilstm,crf层。bilstm对于捕捉序列数据的长依
赖非常有效，而crf层主要是去学习实体间的状态依赖关系，学习到隐状态间的转移概率，拿BIO数据集来说，一个实体一定是BIIII..这种结构，不存在IIII这
种结构，也不存在BOIIIII这种结构，crf会学习到B之后只能接I,I的最前面必须有B.这是softmax层所学习不到的。
# 数据
- 采用人民日报BIO数据`B`表示实体的开头,`I`表示实体的其他部分，`O`表示非实体，`ORG`:表示组织实体，`PER`:表示人名，`LOC`:表示地名，格式如下
```
中 B-ORG
国 I-ORG
致 I-ORG
公 I-ORG
党 I-ORG
十 I-ORG
一 I-ORG
大 I-ORG
的 O
贺 O
词 O

各 O
位 O
代 O
表 O
、 O
各 O
位 O
同 O
志 O
： O

```
`\n\n`分割文档，`\n`分割字符和实体标签，


# 使用

- BILSTM_CRF_ZH_NER：以crf层为最终层的实体识别。

```bash
python BILSTM_CRF_ZH_NER.py
```

# 结果：
## BILSTM_CRF_ZH_NER结果：
- 输入1：
```
中华人民共和国国务院总理周恩来在外交部长陈毅，副部长王东的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚
```
- 输出：
```
PER: 周恩来 陈毅 王东
ORG: 中华人民共和国国务院 外交部
LOC： 埃塞俄比亚 非洲 阿尔巴尼亚
```



# 参考
https://github.com/wenzhongdaipi/Kears-Chinese-Ner


