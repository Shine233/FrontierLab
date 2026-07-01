# 划时代方法（Dataview 动态视图）

```dataview
TABLE year AS 年份, direction AS 方向, supersedes AS 取代, builds_on AS 基于
FROM "papers"
WHERE landmark = true
SORT year DESC
```
