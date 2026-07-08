#  1、安装环境
（1）创建环境
`conda create -n mambavision python==3.10.3`

（2）安装torch、torchvision
`conda activate mambavision pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

（3）安装mmcv
` pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`

我这里之前装了mmseg的环境，就直接复制之前创建的mmseg的环境版本
![[attachments/Pasted image 20260511144442.png]]

（4）安装ssm、causal
这里安装的版本是： causal_conv1d-1.5.0.post5 + mamba_ssm-2.2.3.post2
链接：

| [[Releases · Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases?page=2)]() |
| ---------------------------------------------------------------------------------------------------- |
| [[Releases · Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases?page=2)]() |

`pip install mamba_ssm-2.2.3.post2+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` 
`pip install causal_conv1d-1.5.0.post5+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
![[attachments/Pasted image 20260511150325.png]]
按提示进行版本的降低
![[attachments/Pasted image 20260511150528.png]]

（5）下载了mmsegmentation的后，在通过文件安装依赖，方便后续训练自己数据的时候进行参数和文件配置的修改
![[attachments/Pasted image 20260511150450.png|500]]

 `cd mmsegmentation` 
 `pip install -v -e .`
![[attachments/Pasted image 20260511151227.png]]

（6）安装requirements.txt中指定版本的transformer
`pip install transformers==4.50.0`

（7）安装指定版本的numpy和opencv防止冲突
`$pip install opencv-python==4.9.0.80$` 
`$pip install numpy==1.26.4$` 

（8）按提示安装依赖包timm、ftfy
![[attachments/Pasted image 20260511154727.png]]
![[attachments/Pasted image 20260511154908.png]]
（9）遇到“ModuleNotFoundError: No module named 'pkg_resources'”
![[attachments/Pasted image 20260523104254.png]]
```
pip install setuptools==60.0.0
```
![[attachments/Pasted image 20260523104213.png]]

（10）开始训练自己的数据任务
![[attachments/Pasted image 20260511160213.png]]

