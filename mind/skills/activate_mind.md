# Activate Mind Skill

## Trigger

当用户提到以下任一意图时，执行本 skill：

- 进入 mind 环境
- activate mind
- 切到 mind 环境
- 使用 mind 环境

## Goal

进入 `mind` 项目目录，并加载运行 `mind` 项目所需的 CUDA、cuDNN 和 conda 环境。

## Steps

严格按下面顺序执行：

```bash
cd /home/yang1078/proj/WhoIsWho/mind
module unload cuDNN
module unload CUDA
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
source ~/.bashrc
conda activate mind
```

## Notes

- 每当用户说“进入 mind 环境”，默认就是执行上面的整套流程。
- 先 `unload` 再 `load`，避免沿用旧的 CUDA 或 cuDNN 模块。
- 在新开的 shell 里，先 `source ~/.bashrc`，确保 `conda activate` 可用。
- 如果已经在 `mind` 目录，也仍然按相同步骤处理环境。
