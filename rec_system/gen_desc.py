import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-nano"
ERROR_LOG = "error_log.txt"
OUTPUT_FILE = "filled_error_descriptions.jsonl"
GEN_ERRORS = "gen_errors.log"
RETRY_LIMIT = 3
SLEEP_BETWEEN = 1.0

def load_error_titles(path):
    titles = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading error titles"):
            parts = line.strip().split(" - ", 1)
            if parts and parts[0]:
                titles.add(parts[0])
    return list(titles)

def load_done_titles(path):
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading completed titles"):
                try:
                    rec = json.loads(line)
                    done.add(rec["title"])
                except json.JSONDecodeError:
                    continue
    return done

def generate_description(title):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You're a movie database assistant. Provide a concise 2-3 sentence plot summary."},
            {"role": "user", "content": f"Movie title: \"{title}\". Provide the official theatrical overview."}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

titles = load_error_titles(ERROR_LOG)
done = load_done_titles(OUTPUT_FILE)

assert len(titles) > 0, "No error titles found in the log file."

pending_titles = [title for title in titles if title not in done]
print(f"Total: {len(titles)}, Done: {len(done)}, Pending: {len(pending_titles)}")

with open(OUTPUT_FILE, "a", encoding="utf-8") as fout, \
        open(GEN_ERRORS, "a", encoding="utf-8") as elog:

    for title in tqdm(pending_titles, desc="Generating descriptions"):
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                desc = generate_description(title)
                break
            except Exception as e:
                elog.write(f"{title} - Attempt {attempt} error: {repr(e)}\n")
                time.sleep(2 ** attempt)
        else:
            desc = ""
            elog.write(f"{title} - Failed after {RETRY_LIMIT} attempts\n")

        record = {"title": title, "description": desc}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        
        time.sleep(SLEEP_BETWEEN)

final_done = load_done_titles(OUTPUT_FILE)
assert len(final_done) >= len(done), "Output file should have more entries"
print(f"Complete. Total entries: {len(final_done)}")