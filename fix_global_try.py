from pathlib import Path

src = Path("app.py")
text = src.read_text(encoding="utf-8").splitlines()

out = []
inside_bad_try = False
indent_level = None

for line in text:
    # Global scope'ta girintili try tespiti
    if not inside_bad_try and line.startswith("    try:"):
        inside_bad_try = True
        indent_level = len(line) - len(line.lstrip())
        out.append("# " + line + "  # AUTO-COMMENTED (illegal global try)")
        continue

    if inside_bad_try:
        cur_indent = len(line) - len(line.lstrip())
        # try bloğu bitene kadar yorumla
        if line.strip().startswith("except") or cur_indent > indent_level:
            out.append("# " + line)
            continue
        else:
            inside_bad_try = False
            indent_level = None

    out.append(line)

src.write_text("\n".join(out), encoding="utf-8")
print("DONE: illegal global try blocks commented.")
