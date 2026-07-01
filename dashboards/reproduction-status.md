# 复现状态（Dataview 动态视图）

```dataview
TABLE reproduction AS 复现状态, status AS 阅读状态, github AS 仓库
FROM "papers"
WHERE reproduction
SORT reproduction ASC, file.name ASC
```
