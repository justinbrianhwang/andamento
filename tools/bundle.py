"""Inline local package imports so an experiment can run as one file.

`colab run` uploads a single script into a fresh notebook kernel. Nothing
else from the repo goes with it, and `__file__` is undefined in a notebook
cell, so `from andamento import legality` cannot work there. Rather than
keeping a second copy of the library inside every experiment, this rewrites
`from andamento import X` into the module's own source at send time, leaving
one source of truth on disk.

Usage:
    python tools/bundle.py experiments/sweep_gpu.py -o /tmp/bundled.py
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORT_RE = re.compile(r"^from andamento import (\w+)(?:\s+#.*)?$", re.MULTILINE)
# The sys.path shim only exists to make the import work locally; it relies on
# __file__ and is meaningless once the module is inlined.
PATH_SHIM_RE = re.compile(
    r"^sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)\n",
    re.MULTILINE)


def inline(source):
    source = PATH_SHIM_RE.sub("", source)
    modules = IMPORT_RE.findall(source)
    for name in modules:
        path = os.path.join(ROOT, "andamento", f"{name}.py")
        if not os.path.exists(path):
            raise SystemExit(f"cannot inline unknown module: andamento/{name}.py")
        with open(path, encoding="utf-8") as f:
            body = f.read()
        # Expose the module under its own name so `legality.foo(...)` still
        # resolves, without needing a real package on the remote machine.
        wrapper = (
            f"# --- inlined from andamento/{name}.py by tools/bundle.py ---\n"
            f"import types as _types\n"
            f"{name} = _types.ModuleType({name!r})\n"
            f"exec(compile({body!r}, 'andamento/{name}.py', 'exec'), {name}.__dict__)\n"
            f"# --- end andamento/{name}.py ---"
        )
        # Pass a function, not a string: re.sub interprets backslash escapes in
        # a replacement string, which would turn the \n inside repr(body) back
        # into real newlines and break the literal.
        source = re.sub(rf"^from andamento import {name}.*$", lambda _: wrapper,
                        source, count=1, flags=re.MULTILINE)
    return source, modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    with open(args.script, encoding="utf-8") as f:
        bundled, modules = inline(f.read())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(bundled)

    print(f"bundled {args.script} -> {args.out}"
          f" (inlined: {', '.join(modules) if modules else 'nothing'})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
