# RMBench 数据转换与自标注对接

按照要求提供标注对接：

## 一条完整的数据样例
见samples文件夹
+ lerobot：源数据的一条数据样例（Rmbench的原始数据集）
+ infidata：中间层数据的一条数据样例（我们标注的对象数据）
+ lerobot：最终训练的数据的一条数据样例

此外，conversion文件夹下，记录了如何进行数据转换的pipeline指南和代码，可供参考


## 自标注细节
见annotation文件夹
+ example_output:标注的中间结果，包括一条可视化适配，和整理后得到的memory标注
+ video2tasks文件夹：标注源代码
+ AUTO_ANNOTATION.md：标注代码的运行流程和指南


