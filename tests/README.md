# Tests / Benchmarks

本目录放轻量测试、sanity check、benchmark 脚本或说明。

建议组织：

```text
tests/<direction>/<method>/
├── README.md              # 测试目标、环境、运行命令、期望输出
├── configs/               # 小型配置，可提交
└── scripts/               # 小型脚本，可提交
```

原则：

- 不提交大数据、模型权重、缓存。
- 每个测试说明输入、输出、成功标准、失败解释。
- 若只是复现实验日志，优先放到 `reproductions/`；若是可重复执行的检查，再放到这里。
