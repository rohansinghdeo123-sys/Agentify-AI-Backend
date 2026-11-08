"""Display only supplied scientific fields from structured formula records."""
import re


def _description(value):
    if isinstance(value,str): return value.strip()
    if isinstance(value,(int,float)) and not isinstance(value,bool): return str(value)
    if isinstance(value,list): return '; '.join(filter(None,(_description(item) for item in value)))
    if isinstance(value,dict):
        return '; '.join(f'{key}: {text}' for key,item in value.items() if (text:=_description(item)))
    return ''


def formula_blocks(value):
    if isinstance(value,str):
        return [{'kind':'formula','text':value.strip()}] if value.strip() else []
    if not isinstance(value,dict): return []
    blocks,seen=[],set()
    for field in ['formula','expression','equation','latex','text']:
        text=value.get(field)
        if not isinstance(text,str) or not text.strip() or text.strip() in seen: continue
        text=text.strip();seen.add(text)
        # Source LaTeX stays intact; delimiters let the existing KaTeX renderer
        # display it. Persisted source values and hashes are never rewritten.
        if (field=='latex' or re.search(r'\\(?:frac|dfrac|sqrt|sum|int|begin|mathrm)\b',text)) and not re.search(r'\$|\\[([]',text):
            text=r'\('+text+r'\)'
        blocks.append({'kind':'formula','text':text})
    if not blocks: return []
    for field,kind in [('symbols','terms'),('variables','terms'),('terms','terms'),('units','units'),('conditions','conditions')]:
        if text:=_description(value.get(field)):
            blocks.append({'kind':kind,'text':text})
    return blocks
