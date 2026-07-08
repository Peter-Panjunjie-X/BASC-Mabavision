# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import mamba_vision

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation') # 用于保存离线评估输出预测的目录
    parser.add_argument(
        '--show', action='store_true', help='show prediction results') 
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')   # 保存绘制图像的目录。如果指定，将会自动保存
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')   # 覆盖在使用的配置中的一些设置，xxx=yyy格式的键值对将被合并到配置文件中。如果要覆盖的值是一个列表，它应该像key="[a,b]"或key=a,b。它还允许嵌套列表/元组值，例如key="[(a,b),(c,d)]"。请注意，必要的引号和不允许有空格。')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')  # 测试时间增强
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()



# CUDA_VISIBLE_DEVICES=3 python semantic_segmentation/tools/test.py --config semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov.py --checkpoint ./work_dir/farmland_mamba_small_BASC_lov/iter_104000_7596.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test  --show



# CUDA_VISIBLE_DEVICES=3 python semantic_segmentation/tools/test.py --config  work_dir/farmland_mamba_base_BASC_lov_test/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov_fk.py  --checkpoint ./work_dir/farmland_mamba_small_BASC_lov/iter_104000_7596.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test  --show  --out ./work_dir/farmland_mamba_base_BASC_lov_test/output

# CUDA_VISIBLE_DEVICES=3 python semantic_segmentation/tools/test.py --config  work_dir/farmland_mamba_base_BASC_lov_test/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov_fk.py  --checkpoint ./work_dir/farmland_mamba_small_BASC_lov/iter_38000_7627.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test  --out ./work_dir/farmland_mamba_base_BASC_lov_test/output

# CUDA_VISIBLE_DEVICES=2 python semantic_segmentation/tools/test.py --config  work_dir/farmland_mamba_base_BASC_lov_test/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov_fk.py  --checkpoint ./work_dir/farmland_mamba_small_BASC_lov/iter_20000_7615.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test  --out ./work_dir/farmland_mamba_base_BASC_lov_test/output


# python semantic_segmentation/tools/test.py  work_dir/farmland_mamba_base_BASC_lov_test/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov_fk.py   work_dir/farmland_mamba_small_BASC_lov/iter_20000_7615.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test  --out ./work_dir/farmland_mamba_base_BASC_lov_test/res_out   

# python semantic_segmentation/tools/test.py  semantic_segmentation/configs/mamba_vision/farmland_mamba_vision_160k_ade20k-512x512_small_BASC_lov.py   work_dir/farmland_mamba_small_BASC_lov/iter_36000.pth  --work-dir ./work_dir/farmland_mamba_base_BASC_lov_test65  --out ./work_dir/farmland_mamba_base_BASC_lov_test/res_65 