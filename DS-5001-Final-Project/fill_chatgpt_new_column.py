import os
import time
from pathlib import Path
import pandas as pd
from openai import OpenAI

BASE_INPUT_CSV = "chatgptNew.csv"
OUTPUT_CSV = "chatgptNew_filled.csv"
MODEL = "gpt-5.4-mini"

INPUT_COST_PER_1M = 0.75
OUTPUT_COST_PER_1M = 4.50

MAX_OUTPUT_TOKENS = 400
MAX_BUDGET_USD = 20.00

PROMPT_TEMPLATE = """Answer the following question in a clear and concise paragraph, similar to how a knowledgeable person might respond online. Avoid bullet points and keep the explanation straightforward.

Question: {question}"""

SUCCESS_SLEEP_SECONDS = 0.1
RATE_LIMIT_BASE_SLEEP_SECONDS = 30
CONNECTION_RETRY_SLEEP_SECONDS = 30
MAX_RETRIES_PER_ROW = 100

def estimate_input_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    return input_cost + output_cost

def is_rate_limit_error(msg: str) -> bool:
    msg = msg.lower()
    return "rate limit" in msg or "429" in msg or "rate_limit_exceeded" in msg

def is_connection_error(msg: str) -> bool:
    msg = msg.lower()
    return (
        "connection error" in msg
        or "api connection error" in msg
        or "timeout" in msg
        or "timed out" in msg
        or "temporarily unavailable" in msg
        or "dns" in msg
        or "ssl" in msg
        or "remoteprotocolerror" in msg
    )

def load_dataframe() -> pd.DataFrame:
    output_path = Path(OUTPUT_CSV)
    if output_path.exists():
        print(f"Resuming from existing file: {OUTPUT_CSV}")
        df = pd.read_csv(output_path)
    else:
        print(f"Starting from base file: {BASE_INPUT_CSV}")
        df = pd.read_csv(BASE_INPUT_CSV)

    if "ChatGPT_New" not in df.columns:
        df["ChatGPT_New"] = ""

    return df

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    client = OpenAI(api_key=api_key)
    df = load_dataframe()

    pending_idx = df.index[
        df["ChatGPT_New"].isna() |
        (df["ChatGPT_New"].astype(str).str.strip() == "") |
        (df["ChatGPT_New"].astype(str).str.lower().isin(["nan", "none"]))
    ].tolist()
    running_estimated_cost = 0.0

    print(f"Loaded {len(df)} rows.")
    print(f"Rows remaining: {len(pending_idx)}")
    print(f"Estimated budget cap: ${MAX_BUDGET_USD:.2f}")

    if not pending_idx:
        print("No empty ChatGPT_New rows found. Nothing to do.")
        return

    print(f"First row to process: {pending_idx[0]}")

    for n, idx in enumerate(pending_idx, start=1):
        question = str(df.at[idx, "Question"])
        prompt = PROMPT_TEMPLATE.format(question=question)

        estimated_next_cost = estimate_cost_usd(
            input_tokens=estimate_input_tokens(prompt),
            output_tokens=MAX_OUTPUT_TOKENS
        )

        if running_estimated_cost + estimated_next_cost > MAX_BUDGET_USD:
            print(f"Stopping before row {idx} to stay under the estimated budget cap.")
            break

        retries = 0
        while True:
            try:
                print(f"Starting row {idx} ({n}/{len(pending_idx)})...")

                response = client.responses.create(
                    model=MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                )

                answer = response.output_text.strip()
                df.at[idx, "ChatGPT_New"] = answer

                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", estimate_input_tokens(prompt))
                    output_tokens = getattr(usage, "output_tokens", MAX_OUTPUT_TOKENS)
                else:
                    input_tokens = estimate_input_tokens(prompt)
                    output_tokens = min(MAX_OUTPUT_TOKENS, max(1, len(answer) // 4))

                running_estimated_cost += estimate_cost_usd(input_tokens, output_tokens)
                df.to_csv(OUTPUT_CSV, index=False)

                print(f"Finished row {idx}.")
                print(f"Estimated spend so far this run: ${running_estimated_cost:.4f}")

                if n % 25 == 0:
                    print(f"Completed {n}/{len(pending_idx)} rows in this run...")

                time.sleep(SUCCESS_SLEEP_SECONDS)
                break

            except KeyboardInterrupt:
                print("\nStopped by user. Saving progress before exit...")
                df.to_csv(OUTPUT_CSV, index=False)
                raise

            except Exception as e:
                msg = str(e)
                retries += 1
                df.to_csv(OUTPUT_CSV, index=False)

                if retries > MAX_RETRIES_PER_ROW:
                    print(f"Too many retries on row {idx}. Saving progress and stopping.")
                    print(f"Last error: {e}")
                    return

                if is_rate_limit_error(msg):
                    wait_time = RATE_LIMIT_BASE_SLEEP_SECONDS * retries
                    print(f"Rate limit on row {idx}. Waiting {wait_time}s before retry {retries}/{MAX_RETRIES_PER_ROW}...")
                    time.sleep(wait_time)
                    continue

                if is_connection_error(msg):
                    print(f"Connection issue on row {idx}. Waiting {CONNECTION_RETRY_SLEEP_SECONDS}s before retry {retries}/{MAX_RETRIES_PER_ROW}...")
                    time.sleep(CONNECTION_RETRY_SLEEP_SECONDS)
                    continue

                print(f"Unexpected error on row {idx}: {e}")
                print("Stopping so you can inspect the issue without skipping rows.")
                return

    print(f"Done. Saved to {OUTPUT_CSV}")
    print(f"Estimated total spend this run: ${running_estimated_cost:.4f}")

if __name__ == "__main__":
    main()
