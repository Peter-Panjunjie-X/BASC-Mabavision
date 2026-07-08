# 1、直接找到mambavision网络结构的py文件

![[attachments/Pasted image 20260525162825.png|490]]

# 2、找到forward函数直接把两个模块加进去
找到第一阶段和第四阶段的特征进行判断，然后接在特征的后面进行处理，再将输出的结果一起add到总体特征中
![[attachments/Pasted image 20260525162948.png]]

# 3、改参数文件里面的值
改成新的模型名称——MM_mamba_vision_BASC
![[attachments/Pasted image 20260525163125.png]]

# 4、指定GPU开始训练即可

```
CUDA_VISIBLE_DEVICES=3 python semantic_segmentation/tools/train.py --config semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov.py  --work-dir ./work_dir/farmland_mamba_small_BASC_lov  --amp
```