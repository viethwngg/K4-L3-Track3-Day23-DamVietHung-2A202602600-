"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter
from urllib.parse import urlsplit

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources.json must be a non-empty JSON list of objects"]
    by_n, urls = {}, set()
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"source at index {index} is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"source at index {index} has invalid integer n: {n!r}")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source at index {index} has invalid URL")
        else:
            try:
                valid = bool(urlsplit(url).netloc) and not re.search(r"\s", url)
            except ValueError:
                valid = False
            if not valid:
                problems.append(f"source at index {index} has invalid URL")
            if url in urls:
                problems.append(f"duplicate source URL: {url}")
            urls.add(url)
    if not isinstance(report_text, str):
        return problems + ["report must be text"]
    clean = _without_code(report_text)
    headings = list(re.finditer(r"(?m)^##[ \t]+References[ \t]*\r?$", clean))
    if not headings:
        problems.append("missing ## References heading")
        body, references = clean, ""
    else:
        if len(headings) != 1:
            problems.append("expected exactly one ## References heading")
        heading = headings[0]
        body, references = clean[:heading.start()], clean[heading.end():]
        if re.search(r"(?m)^#{1,6}\s+", references):
            problems.append("References must be the final report section")
    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?![ \t]*\()", body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                first, last = map(int, span.groups())
                if not 0 <= last - first <= 200:
                    problems.append(f"invalid citation range [{part}]")
                    continue
                cited.update(range(first, last + 1))
            else:
                cited.add(int(part))
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")
    counts = Counter()
    for match in re.finditer(r"(?m)^\[(\d+)\][ \t]*(.*)$", references):
        n, line = int(match.group(1)), match.group(2)
        counts[n] += 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        found = []
        for raw in re.findall(r'https?://[^\s<>"`]+', line):
            url = raw.rstrip(".,;").rstrip("]")
            while url.endswith(")") and url.count(")") > url.count("("):
                url = url[:-1]
            found.append(url)
        if len(found) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL (found {len(found)})")
        elif n in by_n and found[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    for n in sorted(by_n):
        if counts[n] == 0:
            problems.append(f"missing reference line [{n}]")
    for n, count in sorted(counts.items()):
        if count > 1:
            problems.append(f"duplicate reference line [{n}]")
    return problems


def _without_code(text):
    """Mask fenced/indented code and inline backtick spans (standard library only)."""
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if re.match(rf"^ {{0,3}}{re.escape(fence[0])}{{{fence[1]},}}[ \t]*\r?\n?$", line):
                fence = None
            lines.append("\n" if line.endswith("\n") else "")
        elif marker:
            fence = (marker[1][0], len(marker[1]))
            lines.append("\n" if line.endswith("\n") else "")
        elif line.startswith(("    ", "\t")):
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(line)
    return re.sub(r"(`+)(?!`)(.*?)\1(?!`)", "", "".join(lines), flags=re.DOTALL)


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
