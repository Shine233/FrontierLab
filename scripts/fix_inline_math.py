"""修复 GitHub 行内公式渲染：在 CJK 与 $...$ 之间插入空格。

用法：python3 scripts/fix_inline_math.py <file.md> [--write]
不加 --write 为 dry-run（只打印将改动的行）。跳过 ``` 代码块与 $$ 展示块。
"""
import re, sys

def fix(text):
    out=[]
    in_code=False
    for line in text.split("\n"):
        st=line.strip()
        if st.startswith("```"):
            in_code=not in_code; out.append(line); continue
        if in_code or st.startswith("$$"):
            out.append(line); continue
        # 对每个行内 $...$（排除 $$），在与 CJK 紧贴处插空格
        def repl(m):
            return "$"+m.group(1)+"$"
        # 先按行内公式切分，重建时补空格
        def process(s):
            res=[]; i=0
            for m in re.finditer(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', s):
                res.append((m.start(), m.end(), m.group(0)))
            if not res: return s
            news=[]; last=0
            for st_,en_,frag in res:
                before=s[st_-1] if st_>0 else ""
                after=s[en_] if en_<len(s) else ""
                pre = " " if (before and before not in " \t([{" and ord(before)>127) else ""
                post= " " if (after and after not in " \t)]}.,;:!?、，。；：！？" and ord(after)>127) else ""
                news.append(s[last:st_]); news.append(pre+frag+post); last=en_
            news.append(s[last:])
            return "".join(news)
        out.append(process(line))
    return "\n".join(out)

p=sys.argv[1]
t=open(p,encoding="utf-8").read()
n=fix(t)
if len(sys.argv)>2 and sys.argv[2]=="--write":
    open(p,"w",encoding="utf-8").write(n); print("wrote",p)
else:
    # dry run: show a diff-ish sample
    import difflib
    d=list(difflib.unified_diff(t.split("\n"),n.split("\n"),lineterm="",n=0))
    print("\n".join(d[:20]))
    print(f"... changed lines: {sum(1 for x in d if x.startswith('+') and not x.startswith('+++'))}")
