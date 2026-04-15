# Connect Shared Tmux Skill

## Trigger

当用户提到以下任一意图时，执行本 skill：

- 连接 shared tmux
- 进入 shared session

## Goal

创建或连接到名为 `shared` 的 tmux session，并在该 session 中按 `activate_mind` 的标准流程进入 `mind` 项目环境；如果当前终端支持交互式 tmux，再连接进去。进入 `shared` 之后，默认将当前会话后续命令视为在 `shared` 环境中执行。

## Steps

严格按下面顺序执行：

```bash
tmux has-session -t shared 2>/dev/null || tmux new-session -d -s shared
```

如果当前终端支持交互式 tmux，再执行：

```bash
tmux attach -t shared
```

## Notes

- 如果 `shared` session 不存在，先创建，再继续后续步骤。
