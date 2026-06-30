# 时间线（Dataview 动态视图）

> 在 Obsidian 中打开本页（需 Dataview 插件）。GitHub 上仅显示查询代码。

```dataview
TABLE year AS 年份, direction AS 方向, key_idea AS 核心, landmark AS 划时代
FROM "papers"
WHERE year
SORT year DESC, file.name ASC
```
