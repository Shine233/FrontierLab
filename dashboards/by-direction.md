# 按方向分组（Dataview 动态视图）

```dataview
TABLE rows.file.link AS 论文, rows.year AS 年份
FROM "papers"
WHERE direction
FLATTEN direction
GROUP BY direction
SORT direction ASC
```
