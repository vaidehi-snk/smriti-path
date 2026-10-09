import json, re, sys, requests

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "stops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "place": {"type": "string"},
                    "then": {"type": "string"},
                    "quote": {"type": "string"},
                    "ask_here": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["place", "then", "quote", "ask_here"],
            },
        },
    },
    "required": ["title", "stops"],
}

SYSTEM = """You turn a spoken memory into a walking tour.
Rules:
- Use ONLY what the speaker said. Never invent places, dates or facts.
- Make one stop for each physical place the speaker mentions, in the order mentioned. Never merge or repeat places.
- "place": a short descriptive name that includes the type of place, never a single bare word.
- "then": 1-2 plain sentences, only facts the speaker gave about THAT place.
- "quote": words copied exactly from the transcript. Never add a label or prefix.
- "ask_here": 2 short, warm questions to ask the speaker while standing there. One must ask what is different about this spot today.

Example transcript: "Near the school gate there was a mango tree and the watchman sold pickles under it. Then we crossed the railway line to the river."
Example stops: [
 {"place": "Mango tree near the school gate", "then": "The watchman sold pickles under a mango tree near the school gate.", "quote": "the watchman sold pickles under it", "ask_here": ["What do you remember about those pickles?", "What is here today instead of the tree?"]},
 {"place": "Railway line crossing", "then": "They crossed the railway line to reach the river.", "quote": "we crossed the railway line to the river", "ask_here": ["How did you cross the line back then?", "What does this crossing look like today?"]}
]"""

def tokens(s):
    return re.sub(r"[^\w\s]", "", s.lower()).split()

def extract(transcript: str) -> dict:
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": transcript},
        ],
    }, timeout=300)
    r.raise_for_status()
    data = json.loads(r.json()["message"]["content"])

    haystack = " ".join(tokens(transcript))
    for stop in data["stops"]:
        stop["quote"] = re.sub(r"^\s*quote\s*:\s*", "", stop["quote"], flags=re.I).strip()
        stop["quote_verified"] = " ".join(tokens(stop["quote"])) in haystack
    return data

if __name__ == "__main__":
    text = open(sys.argv[1], encoding="utf-8").read()
    result = extract(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = sum(s["quote_verified"] for s in result["stops"])
    print(f"\nVerified quotes: {ok}/{len(result['stops'])}")