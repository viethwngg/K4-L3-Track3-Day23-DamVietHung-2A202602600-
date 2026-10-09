"""Normalize fixed template headings INSIDE the sandbox, before citation finalization.

This changes heading formatting only, never evidence, citations, or source metadata.
"""
import re
import sys
from pathlib import Path

REPORT = '/tmp/work/report/report.md'
HEADINGS = {'tl;dr': 'TL;DR', 'tldr': 'TL;DR', 'background': 'Background',
            'trends and open problems': 'Trends and open problems', 'references': 'References'}


def normalize(text):
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            lines.append(line)
            if re.match(rf'^ {{0,3}}{re.escape(fence[0])}{{{fence[1]},}}[ \t]*\r?\n?$', line):
                fence = None
            continue
        if marker:
            fence = (marker[1][0], len(marker[1]))
            lines.append(line)
            continue
        heading = re.match(r'^#{1,3}[ \t]+(.*?)[ \t]*\r?\n?$', line)
        key = heading[1].strip(' *_`').rstrip(':').casefold() if heading else ''
        if key in HEADINGS:
            ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
            line = '## ' + HEADINGS[key] + ending
        lines.append(line)
    return ''.join(lines)


def main(argv):
    path = Path(argv[1] if len(argv) > 1 else REPORT)
    try:
        original = path.read_text(encoding='utf-8')
        updated = normalize(original)
        if updated != original:
            path.write_text(updated, encoding='utf-8')
        print('NORMALIZED: fixed template headings checked')
        return 0
    except OSError as exc:
        print(f'cannot normalize report: {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
