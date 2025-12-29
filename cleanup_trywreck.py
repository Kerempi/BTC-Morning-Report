from pathlib import Path

SRC = Path("app.py")
BAK = Path("app.py.bak_before_cleanup")

text = SRC.read_text(encoding="utf-8").splitlines()

BAK.write_text("\n".join(text), encoding="utf-8")
print(f"Backup wrote: {BAK}")

out = []
i = 0
n = len(text)

def lstrip_len(s: str) -> int:
    return len(s) - len(s.lstrip(" \t"))

while i < n:
    line = text[i]
    s = line.lstrip()

    # Case A: commented-out try marker we used before -> comment everything that looks like its body
    if s.startswith("#") and "AUTO-COMMENTED (illegal global try)" in s:
        out.append(line)  # keep marker
        i += 1
        # comment subsequent indented lines until we hit a blank line or a new top-level statement
        while i < n:
            nxt = text[i]
            ns = nxt.lstrip()
            if ns.startswith("except") or ns.startswith("except "):
                out.append("# " + nxt)
                i += 1
                continue

            # stop on def/class/with/if/for/return at same or lower indent than the marker block was meant to cover
            if ns.startswith(("def ", "class ", "with ", "if ", "for ", "while ", "return ", "import ", "from ")):
                break

            # if line is already commented, keep
            if ns.startswith("#"):
                out.append(nxt)
            else:
                # comment anything that is indented (likely orphaned body) or looks like a continuation
                if lstrip_len(nxt) > 0 or ns.startswith((")", "]", "}", ",")):
                    out.append("# " + nxt)
                else:
                    # top-level non-indented statement -> stop
                    break
            i += 1
        continue

    # Case B: orphan except at any indent -> comment it and its body
    if s.startswith("except"):
        out.append("# " + line + "  # AUTO-COMMENTED (orphan except)")
        i += 1
        # comment following indented lines (handler body)
        while i < n and lstrip_len(text[i]) > lstrip_len(line):
            out.append("# " + text[i])
            i += 1
        continue

    out.append(line)
    i += 1

SRC.write_text("\n".join(out), encoding="utf-8")
print("DONE: cleanup applied to app.py")
