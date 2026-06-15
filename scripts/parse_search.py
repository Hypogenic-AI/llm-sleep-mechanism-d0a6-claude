import json, sys

def load_json_loose(path):
    txt = open(path).read()
    # find first '{' and parse the first JSON object
    start = txt.find('{')
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(txt[start:])
        return obj
    except Exception as e:
        return None

for path in sys.argv[1:]:
    d = load_json_loose(path)
    print("####", path, "TOTAL", (d or {}).get('total'))
    if not d:
        print("  <no json>")
        continue
    for p in d.get('papers', []):
        if p.get('relevance', 0) >= 2:
            print(p.get('relevance'), '|', p.get('year'), '|', p.get('title'), '|', p.get('url'), '|cit', p.get('citations'))
