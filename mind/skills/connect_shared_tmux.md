# Connect Shared Tmux Skill

## Trigger

当用户提到以下任一意图时，执行本 skill：

- 连接 shared tmux
- 进入 shared session
- 打开 shared tmux
- 创建 shared tmux
- 用 shared 跑 mind

## Goal

创建或连接到名为 `shared` 的 tmux session，并在该 session 中按 `activate_mind` 的标准流程进入 `mind` 项目环境；如果当前终端支持交互式 tmux，再连接进去。进入 `shared` 之后，默认将当前会话后续命令视为在 `shared` 环境中执行。

## Steps

严格按下面顺序执行：

```bash
tmux has-session -t shared 2>/dev/null || tmux new-session -d -s shared
tmux send-keys -t shared:0.0 "cd /home/yang1078/proj/WhoIsWho/mind" C-m
tmux send-keys -t shared:0.0 "module unload cuDNN" C-m
tmux send-keys -t shared:0.0 "module unload CUDA" C-m
tmux send-keys -t shared:0.0 "module load CUDA/12.1.1" C-m
tmux send-keys -t shared:0.0 "module load cuDNN/8.9.2.26-CUDA-12.1.1" C-m
tmux send-keys -t shared:0.0 "source ~/.bashrc" C-m
tmux send-keys -t shared:0.0 "conda activate mind" C-m
```

如果当前终端支持交互式 tmux，再执行：

```bash
tmux attach -t shared
```

## Notes

- 如果 `shared` session 不存在，先创建，再继续后续步骤。
- 环境激活步骤需要与 `activate_mind` 保持一致，不要省略 `module unload`、`module load` 或 `source ~/.bashrc`。
- 默认在 `shared:0.0` 窗口执行命令；如果该 session 已被其他人使用，执行前先留意当前窗口内容，避免打断已有任务。
- 有些非交互式终端或受限终端不能直接执行 `tmux attach -t shared`，这时先完成 session 创建和环境激活，再提示用户在自己的终端里手动执行 `tmux attach -t shared`。
- 一旦用户说“进入 shared tmux”或等价表达，后续当前会话中的命令默认都应在 `shared` session 中执行，除非用户明确要求退出 `shared`、切回普通 shell，或指定在别的会话中执行。
