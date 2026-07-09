"""修复 GitHub 行内公式渲染问题。

两类修复：
1. CJK-空格：在中文与 $...$ 之间插入 ASCII 空格（GitHub 要求 $ 前后为空格/标点才识别为公式）。
2. 下划线转义：当同一行有 >=2 个含下划线的行内公式时，GitHub 会把这些 `_` 两两配对当成
   斜体强调标记，吃掉下划线并破坏公式。对这些行，把行内公式内的未转义 `_` 改成 `\_`
   （GitHub 与 Obsidian 两端均正确渲染为下标）。

用法：python3 scripts/fix_inline_math.py <file.md> [--write]
不加 --write 为 dry-run（打印将改动的行数与样例）。跳过 ``` 代码块与 $$ 展示块。
"""
import re, sys

INLINE = re.compile(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)')

def escape_underscores(frag):
    # frag 形如 "$...$"；把内部未转义的 _ 变成 \_
    inner = frag[1:-1]
    inner = re.sub(r'(?<!\\)_', r'\\_', inner)
    return "$" + inner + "$"

def process_line(s):
    spans = [(m.start(), m.end(), m.group(0)) for m in INLINE.finditer(s)]
    if not spans:
        return s
    # 是否触发下划线转义：>=2 个行内公式含下划线
    with_us = [f for _, _, f in spans if '_' in f]
    do_escape = len(with_us) >= 2
    news = []; last = 0
    for st_, en_, frag in spans:
        before = s[st_-1] if st_ > 0 else ""
        after = s[en_] if en_ < len(s) else ""
        pre = " " if (before and before not in " \t([{" and ord(before) > 127) else ""
        post = " " if (after and after not in " \t)]}.,;:!?、，。；：！？" and ord(after) > 127) else ""
        if do_escape and '_' in frag:
            frag = escape_underscores(frag)
        news.append(s[last:st_]); news.append(pre + frag + post); last = en_
    news.append(s[last:])
    return "".join(news)

def fix(text):
    out = []; in_code = False
    for line in text.split("\n"):
        st = line.strip()
        if st.startswith("```"):
            in_code = not in_code; out.append(line); continue
        if in_code or st.startswith("$$"):
            out.append(line); continue
        out.append(process_line(line))
    return "\n".join(out)

if __name__ == "__main__":
    p = sys.argv[1]
    t = open(p, encoding="utf-8").read()
    n = fix(t)
    if len(sys.argv) > 2 and sys.argv[2] == "--write":
        open(p, "w", encoding="utf-8").write(n)
        print("wrote", p)
    else:
        import difflib
        d = list(difflib.unified_diff(t.split("\n"), n.split("\n"), lineterm="", n=0))
        print("\n".join(d[:24]))
        print(f"... changed lines: {sum(1 for x in d if x.startswith('+') and not x.startswith('+++'))}")
