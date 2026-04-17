import os
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "chatgptNew.csv"
OUTPUT_CSV = "chatgptNew_filled.csv"
MODEL = "gpt-5.4-mini"

INPUT_COST_PER_1M = 0.75
OUTPUT_COST_PER_1M = 4.50

MAX_OUTPUT_TOKENS = 400
MAX_BUDGET_USD = 20.00

PROMPT_TEMPLATE = """Answer the following question in a clear and concise paragraph, similar to how a knowledgeable person might respond online. Avoid bullet points and keep the explanation straightforward.

Question: {question}"""

def estimate_input_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    return input_cost + output_cost

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    client = OpenAI(api_key=api_key)
    df = pd.read_csv(INPUT_CSV)

    if "ChatGPT_New" not in df.columns:
        df["ChatGPT_New"] = ""

    pending_idx = df.index[df["ChatGPT_New"].fillna("").astype(str).str.strip() == ""].tolist()

    running_estimated_cost = 0.0

    print(f"Loaded {len(df)} rows.")
    print(f"Rows remaining: {len(pending_idx)}")
    print(f"Estimated budget cap: ${MAX_BUDGET_USD:.2f}")

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

        try:
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

            if n % 50 == 0:
                print(f"Completed {n}/{len(pending_idx)} rows...")
                print(f"Estimated spend so far: ${running_estimated_cost:.4f}")

            time.sleep(0.2)

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            df.to_csv(OUTPUT_CSV, index=False)
            time.sleep(2)

    print(f"Done or paused. Saved to {OUTPUT_CSV}")
    print(f"Estimated total spend: ${running_estimated_cost:.4f}")

if __name__ == "__main__":
    main()
