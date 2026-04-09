# Check GPU Status Skill

## Trigger

当用户提到以下任一意图时，执行本 skill：

- 查看 gpu 情况
- 查看GPU情况
- 查看集群 gpu 占用情况
- 看一下 gpu 占用
- 查询集群 gpu 资源
- 哪些 gpu 节点空闲

## Goal

查看当前集群各个节点的 GPU 占用情况，重点确认哪些节点处于 `mix`、`alloc` 或 `down` 状态，以及每个 GPU 节点已分配了多少 GPU 资源。

## Steps

严格按下面顺序执行：

```bash
sinfo -N
scontrol show nodes | awk '
/NodeName=/ {
  node = ""; state = ""; gres = ""; alloc = ""
  for (i = 1; i <= NF; i++) {
    if ($i ~ /^NodeName=/) { split($i, a, "="); node = a[2] }
    if ($i ~ /^State=/) { split($i, a, "="); state = a[2] }
    if ($i ~ /^Gres=/) { split($i, a, "="); gres = a[2] }
    if ($i ~ /^AllocTRES=/) {
      alloc = substr($i, 11)
      for (j = i + 1; j <= NF; j++) {
        if ($j ~ /^[A-Za-z]+=/) break
        alloc = alloc " " $j
      }
    }
  }
  if (node ~ /^gpu-/) {
    print node " | " state " | " gres " | AllocTRES=" alloc
  }
}'
```

## Notes

- 先执行 `sinfo -N`，快速看所有节点状态。
- 再执行 `scontrol show nodes` 的解析命令，查看每个 GPU 节点的 `Gres` 和 `AllocTRES`。
- 如果用户只是想知道“有没有空闲卡”，优先关注 `mix` 状态节点，这类节点通常表示还有部分资源可用。
- 如果节点是 `alloc`，通常表示该节点 GPU 已被全部占用。
- 如果节点是 `down` 或 `not responding`，直接提示该节点当前不可用。
