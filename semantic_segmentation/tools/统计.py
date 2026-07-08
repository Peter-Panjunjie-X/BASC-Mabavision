import os
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmseg.registry import DATASETS
from PIL import Image
from tqdm import tqdm

# ====== 改成你自己的配置文件路径 ======
config_path = '/mnt/d/seg_dir/mambavision_main/mmsegmentation/configs/_base_/datasets/break_ground.py'

cfg = Config.fromfile(config_path)

# 初始化 registry
init_default_scope('mmseg')

# 构建训练集
dataset = DATASETS.build(cfg.train_dataloader.dataset)

bg_count = 0
fg_count = 0
total_images = len(dataset)

print(f"总图片数: {total_images}")

for i in tqdm(range(total_images)):
    data_info = dataset.get_data_info(i)

    # 获取标注文件路径
    seg_map_path = data_info['seg_map_path']

    # 读取标注（灰度图，像素值就是类别 id）
    seg = np.array(Image.open(seg_map_path))

    bg_count += np.sum(seg == 0)
    fg_count += np.sum(seg == 1)

total = bg_count + fg_count
print(f"\n===== 类别像素统计 =====")
print(f"background:  {bg_count:>12,} 像素 ({bg_count / total * 100:.2f}%)")
print(f"break_ground:{fg_count:>12,} 像素 ({fg_count / total * 100:.2f}%)")
print(f"ratio        : 1 : {bg_count / fg_count:.1f}")

# 根据逆频率算建议权重
w0 = total / (2 * bg_count)
w1 = total / (2 * fg_count)
print(f"\n===== 建议的 class_weight =====")
print(f"class_weight = [{w0:.4f}, {w1:.4f}]")