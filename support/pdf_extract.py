import re
from collections import Counter

import pymupdf


EXTENSION = {
    "Z": "∫", "R": "∫", "X": "∑", "P": "∑", "Y": "∏", "Q": "∏",
    "H": "∮", "I": "∮", "p": "√", "q": "√", "r": "√", "s": "√",
    "\x00": "(", "\x01": ")", "\x02": "[", "\x03": "]",
    "\x08": "{", "\x09": "}", "\x10": "(", "\x11": ")",
    "\x12": "(", "\x13": ")", "\x14": "[", "\x15": "]",
    "\x1a": "{", "\x1b": "}", "\x0c": "|", "n": "{", "o": "}",
    "\uf8eb": "(", "\uf8ed": "(", "\uf8ec": "",
    "\uf8f6": ")", "\uf8f8": ")", "\uf8f7": "",
    "\uf8f1": "{", "\uf8f2": "", "\uf8f3": "}",
    "\uf8fc": "}", "\uf8fd": "", "\uf8fe": "{", "\uf8f4": "|",
}


def decode_char(char, font):
    if font.startswith("F") and font[1:].isdigit():
        if 192 <= ord(char) <= 255 or char in "¨¸":
            char = char.encode("latin1").decode("cp1251")
        return {"\x16": "—", "\x1f": "fi", "\x9f": "§"}.get(char, char)
    if "CMEX" in font:
        return EXTENSION.get(char, char)
    if "MSBM" in font:
        return {"C": "ℂ", "R": "ℝ", "Z": "ℤ", "N": "ℕ"}.get(char, char)
    return char


def glyphs(block):
    chars = []
    for line in block["lines"]:
        for span in line["spans"]:
            for char in span["chars"]:
                value = decode_char(char["c"], span["font"])
                if value.strip():
                    y = char["origin"][1]
                    if "CMEX" in span["font"]:
                        offset = {"Z": 1.36, "R": 0.35, "X": 0.75, "P": 0.3,
                                  "Y": 0.75, "Q": 0.3, "\x12": 1.4, "\x13": 1.4,
                                  "\x10": 1.85, "\x11": 1.85}.get(char["c"], 0.8)
                        y += offset * span["size"]
                    chars.append({"text": value, "x": char["bbox"][0],
                                  "right": char["bbox"][2], "y": y,
                                  "size": span["size"], "font": span["font"],
                                  "math": not span["font"].startswith("F")})
    return chars


def horizontal_rules(page):
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                a, b = item[1:3]
                if abs(a.y - b.y) < 0.2 and abs(a.x - b.x) > 1:
                    rules.append((min(a.x, b.x), max(a.x, b.x), a.y))
    return sorted(set(rules), key=lambda r: r[1] - r[0])


def inline(chars):
    chars = sorted(chars, key=lambda c: c["x"])
    if not chars:
        return ""
    full = [c for c in chars if c["size"] >= 9]
    baseline = Counter(round(c["y"], 1) for c in (full or chars)).most_common(1)[0][0]
    result = ""
    previous = None
    script = None
    for char in chars:
        shift = char["y"] - baseline
        current = None
        if char["math"] and char["size"] < 8.5 and abs(shift) > 1.2:
            current = "^" if shift < 0 else "_"
        if current != script:
            if script:
                result += "}"
            if current:
                result += current + "{"
        if previous and current is None:
            if char["x"] - previous["right"] > 1.35:
                result += " "
        result += char["text"]
        previous, script = char, current
    if script:
        result += "}"
    return result


def recover_math(chars, rules):
    used = set()
    for left, right, y in rules:
        for char in chars:
            if (("CMMI" in char["font"] or "MSBM" in char["font"]) and len(char["text"]) == 1
                    and char["text"].isalpha() and abs(char["x"] - left) < 0.7
                    and abs(char["right"] - right) < 1.2
                    and 0.42 * char["size"] < char["y"] - y < 0.89 * char["size"]):
                char["text"] = "\\bar{" + char["text"] + "}"
                used.add((left, right, y))
                break
    for left, right, y in rules:
        if (left, right, y) in used:
            continue
        inside = [c for c in chars if c["math"]
                  and left - 0.5 <= (c["x"] + c["right"]) / 2 <= right + 0.5]
        if any(1 < y - c["y"] < 10 for c in inside):
            continue
        bases = [c for c in inside if c["size"] >= 9 and 3 < c["y"] - y < 16]
        if not bases or any("\\bar{" in c["text"] for c in bases):
            continue
        baseline = min(c["y"] for c in bases)
        members = [c for c in inside if baseline - 7 <= c["y"] <= baseline + 3]
        if abs(min(c["x"] for c in members) - left) > 1 or abs(max(c["right"] for c in members) - right) > 2:
            continue
        ids = {id(c) for c in members}
        root = next((c for c in chars if c["text"] == "√" and 0 < left - c["x"] < 16
                     and abs(c["y"] - baseline) < 18), None)
        text = "\\overline{" + inline(members) + "}"
        if root:
            indices = [c for c in chars if c["math"] and c["size"] < 8.5
                       and root["x"] - 7 < c["x"] < left and 3.5 < baseline - c["y"] < 18]
            text = "\\sqrt" + ("[" + inline(indices) + "]" if indices else "") + "{" + inline(members) + "}"
            ids.update(id(c) for c in indices + [root])
        chars = [c for c in chars if id(c) not in ids]
        chars.append({"text": text, "x": root["x"] if root else left,
                      "right": right, "y": baseline, "size": 10, "font": "overline", "math": True})
        used.add((left, right, y))
    for left, right, y in rules:
        if (left, right, y) in used:
            continue
        candidates = [c for c in chars if c["math"]
                      and left - 0.5 <= (c["x"] + c["right"]) / 2 <= right + 0.5]
        above = [c for c in candidates if 1.5 < y - c["y"] < 16]
        below = [c for c in candidates if 3 < c["y"] - y < 18]
        if not above or not below:
            continue
        ay = max(c["y"] for c in above)
        by = min(c["y"] for c in below)
        if not (2 < y - ay < 9 and 6 < by - y < 15):
            continue
        above = [c for c in above if abs(c["y"] - ay) < 4.5]
        below = [c for c in below if abs(c["y"] - by) < 4.5]
        members = {id(c) for c in above + below}
        chars = [c for c in chars if id(c) not in members]
        chars.append({"text": "\\frac{" + inline(above) + "}{" + inline(below) + "}",
                      "x": left, "right": right, "y": y + 2.5, "size": 10,
                      "font": "fraction", "math": True})
    return chars


def block_text(block, rules):
    chars = recover_math(glyphs(block), rules)
    removed = set()
    for operator in chars:
        if operator["text"] not in ("∫", "∑", "∏", "∮"):
            continue
        for side, sign in (("_", 1), ("^", -1)):
            limits = [c for c in chars if c["math"] and c["size"] < 8.5
                      and operator["x"] - 5 < c["x"] < operator["right"] + 5
                      and 5 < sign * (c["y"] - operator["y"]) < 22]
            if limits:
                operator["text"] += side + "{" + inline(limits) + "}"
                removed.update(id(c) for c in limits)
    chars = [c for c in chars if id(c) not in removed]
    anchors = []
    for char in sorted(chars, key=lambda c: c["y"]):
        if char["size"] >= 9 and "CMEX" not in char["font"]:
            if not anchors or abs(anchors[-1] - char["y"]) > 3:
                anchors.append(char["y"])
    rows = [[y, []] for y in anchors]
    for char in chars:
        if rows:
            min(rows, key=lambda r: abs(r[0] - char["y"]))[1].append(char)
        else:
            rows.append([char["y"], [char]])
    text = "\n".join(inline(row) for _, row in rows)
    text = re.sub(r"([а-яё])[-]\n([а-яё])", r"\1\2", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


def extract_pdf(path):
    corpus = []
    section = "Начало книги"
    with pymupdf.open(path) as document:
        for page_no, page in enumerate(document, 1):
            rules = horizontal_rules(page)
            groups = []
            for block in page.get_text("rawdict")["blocks"]:
                if "lines" not in block or block["bbox"][3] < 42:
                    continue
                plain = "".join(c["text"] for c in glyphs(block))
                prose = len(re.findall("[А-Яа-яЁё]", plain)) >= 8
                new_paragraph = prose and (plain[0].isupper() or plain.startswith("§"))
                if not groups or new_paragraph:
                    groups.append({"lines": list(block["lines"])})
                else:
                    groups[-1]["lines"].extend(block["lines"])
            for ordinal, block in enumerate(groups, 1):
                text = block_text(block, rules)
                if not text:
                    continue
                heading = re.match(r"§\s*\d+\.\s*[^\n]+", text)
                if heading:
                    section = heading.group()
                corpus.append({"block_id": f"b_{page_no:03d}_{ordinal:03d}",
                               "page": page_no, "section": section, "text": text})
    return corpus
