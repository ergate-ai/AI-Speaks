import os
import re
import requests
import uuid
import json
from datetime import datetime, timezone
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

SESSION_ID = uuid.uuid4()

MODEL_NAME = "mistral"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

PROMPT_FILE = "prompt.txt"
DB_FILE = "./mind/mind.csv"


class MyOllamaLLM(OllamaLLM):
    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: list = None) -> str:
        payload = {"model": MODEL_NAME, "prompt": prompt}
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        response.raise_for_status()
        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                token_data = json.loads(line)
            except Exception as e:
                print("Error parsing line:", line, e)
                continue
            full_text += token_data.get("response", "")
            if (
                token_data.get("done", False)
                and token_data.get("done_reason", "") == "stop"
            ):
                break
        return full_text


def extract_csv_line(response_text: str) -> str:
    pattern = r"\[START\](.*?)\[END\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        current_timestamp = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        return current_timestamp, extracted
    return None


def extract_story(response_text: str) -> str:
    pattern = r"\[START_STORY\](.*?)\[END_STORY\]"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        return extracted
    return None


def read_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return f.read().strip()
    return


def update_prompt_file(new_prompt):
    if new_prompt and isinstance(new_prompt, str):
        with open(PROMPT_FILE, "w") as f:
            f.write(new_prompt)
        print("Prompt file updated.")


def generate_initial_csv_langchain(
    prompt_instructions: str,
    *,
    current_date: str,
    events_to_avoid: list[str],
) -> list:
    """Generate the first mind entry while avoiding previously used events."""
    prompt_text = prompt_instructions.format(
        current_date=current_date,
        events_to_avoid=", ".join(events_to_avoid) if events_to_avoid else ""
    )
    llm = MyOllamaLLM(model=MODEL_NAME)
    prompt_template = PromptTemplate(input_variables=[], template=prompt_text)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({})
    print("Initial CSV response:", response)
    timestamp, csv_line = extract_csv_line(response)
    story = extract_story(response)

    if csv_line and story:
        return (
            f"{humanize(timestamp)},{SESSION_ID},{csv_line}",
            f"Title: {csv_line}\n\nGenerated on: {humanize(timestamp)}\n\n{story}",
        )


def humanize(d):
    try:
        return datetime.fromisoformat(d).strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        pass


def extract_event_date(title: str) -> str | None:
    """Return a YYYY-MM-DD date string from the title if present."""
    if not title:
        return None
    match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", title)
    if match:
        return match.group(0)
    match = re.search(r"\b\d{4}/\d{2}/\d{2}\b", title)
    if match:
        return match.group(0)
    return None


def extract_event_name(title: str) -> str | None:
    """Return the event name preceding the date portion of the title."""
    if not title:
        return None
    title = title.split(",", 1)[0].strip()
    match = re.search(r"^(.*?)\s*-\s*\d{4}[/-]\d{2}[/-]\d{2}", title)
    if match:
        return match.group(1).strip()
    return None


def collect_used_events(lines: list[str]) -> list[str]:
    """Return a list of previously referenced historical event names."""
    events: list[str] = []
    for line in lines:
        parts = line.split(",", 2)
        if len(parts) < 3:
            continue
        title = parts[2].strip()
        event = extract_event_name(title)
        if event and event not in events:
            events.append(event)
    return events


def extend_csv_langchain(
    existing_csv: list,
    prompt_instructions: str,
    *,
    current_date: str,
    events_to_avoid: list[str],
) -> list:
    """Extend the mind while ensuring new events are unique."""
    prompt_text = prompt_instructions.format(
        current_date=current_date,
        events_to_avoid=", ".join(events_to_avoid) if events_to_avoid else ""
    )
    llm = MyOllamaLLM(model=MODEL_NAME)
    current_csv_str = "\n".join(existing_csv)
    prompt_template = PromptTemplate(
        input_variables=["existing_csv"],
        template=(
            "Given the following CSV mind entries (each line is in the format [START]timestamp,title[END]):\n\n"
            "{existing_csv}\n\n"
            f"{prompt_text}"
        ),
    )
    formatted_prompt = prompt_template.format(existing_csv=current_csv_str)
    response = llm._call(formatted_prompt)
    print("Extended CSV response:", response)
    timestamp, csv_line = extract_csv_line(response)
    story = extract_story(response)

    if csv_line and story:
        return (
            f"{humanize(timestamp)},{SESSION_ID},{csv_line}",
            f"Title: {csv_line}\n\nGenerated on: {humanize(timestamp)}\n\n{story}",
        )


def update_db():
    prompt_instructions = read_prompt()
    if not prompt_instructions:
        return
    current_date = datetime.now(timezone.utc).strftime("%B %d")

    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                existing_data = [line.strip() for line in f if line.strip()]
        except Exception:
            existing_data = []
        events = collect_used_events(existing_data)
        mind, story = extend_csv_langchain(
            existing_data,
            prompt_instructions,
            current_date=current_date,
            events_to_avoid=events,
        )
        updated_data = existing_data + [mind]
    else:
        events: list[str] = []
        mind, story = generate_initial_csv_langchain(
            prompt_instructions,
            current_date=current_date,
            events_to_avoid=events,
        )
        updated_data = [mind]


    os.makedirs("mind", exist_ok=True)

    story_dir = os.path.join("mind", "stories")
    os.makedirs(story_dir, exist_ok=True)
    story_file_path = os.path.join(story_dir, f"{SESSION_ID}.txt")

    with open(DB_FILE, "w") as f:
        for line in updated_data:
            f.write(line + "\n")

    with open(story_file_path, "w") as story_file:
        story_file.write(story)

    return updated_data


def main():
    if update_db():
        print("Mind CSV updated successfully")
    else:
        print("Mind CSV NOT updated")


if __name__ == "__main__":
    main()
