#!/usr/bin/env python3
"""
llm_batch_processor.py

Batch-process prompts from a CSV using the OpenAI Responses API (OpenAI Python >=1.0).

Usage (example):
    python3 llm_batch_processor.py \
        --input prompts.csv \
        --output responses.csv \
        --system system_prompt.txt \
        --model gpt-4o \
        --concurrency 3 \
        --max-output-tokens 1024 \
        --save-interval 5 \
        --price-input-per-1k 0.0025 \
        --price-output-per-1k 0.01

Features:
 - Modes:
    * all           : process all rows from prompts.csv
    * continue      : skip IDs already present in output CSV and process remaining
    * rerun_failed  : find rows in the output CSV with status error/refused and rerun those (append results)
    * rerun_ids     : rerun a user-specified list of IDs (comma-separated)
 - Concurrency with ThreadPoolExecutor
 - Resilient: errors/refusals are recorded and processing continues
 - Cost/token tracking (optional, via CLI prices)
 - Appends rerun results (C-1 behaviour); does not overwrite original rows
 - No timestamp column (per user request)

Notes:
- The script will read OPENAI_API_KEY from the environment.
- To enable cost computations, provide --price-input-per-1k and/or --price-output-per-1k
  (USD per 1000 tokens). Default is 0.0 (costs not calculated).
- The script is resilient: if a single row fails, it logs the error in the output CSV and continues.
- Current OpenAI model prices (per 1M tokens):
Model	Input	Cached input	Output
gpt-5	$1.25	$0.125	$10.00
gpt-5-mini	$0.25	$0.025	$2.00
gpt-5-nano	$0.05	$0.005	$0.40
gpt-5-chat-latest	$1.25	$0.125	$10.00
gpt-5-codex	$1.25	$0.125	$10.00
gpt-5-pro	$15.00	-	$120.00
gpt-4.1	$2.00	$0.50	$8.00
gpt-4.1-mini	$0.40	$0.10	$1.60
gpt-4.1-nano	$0.10	$0.025	$0.40
gpt-4o	$2.50	$1.25	$10.00
gpt-4o-2024-05-13	$5.00	-	$15.00
gpt-4o-mini	$0.15	$0.075	$0.60
gpt-realtime	$4.00	$0.40	$16.00
gpt-realtime-mini	$0.60	$0.06	$2.40
gpt-4o-realtime-preview	$5.00	$2.50	$20.00
gpt-4o-mini-realtime-preview	$0.60	$0.30	$2.40
gpt-audio	$2.50	-	$10.00
gpt-audio-mini	$0.60	-	$2.40
gpt-4o-audio-preview	$2.50	-	$10.00
gpt-4o-mini-audio-preview	$0.15	-	$0.60
o1	$15.00	$7.50	$60.00
o1-pro	$150.00	-	$600.00
o3-pro	$20.00	-	$80.00
o3	$2.00	$0.50	$8.00
o3-deep-research	$10.00	$2.50	$40.00
o4-mini	$1.10	$0.275	$4.40
o4-mini-deep-research	$2.00	$0.50	$8.00
o3-mini	$1.10	$0.55	$4.40
o1-mini	$1.10	$0.55	$4.40
codex-mini-latest	$1.50	$0.375	$6.00
gpt-5-search-api	$1.25	$0.125	$10.00
gpt-4o-mini-search-preview	$0.15	-	$0.60
gpt-4o-search-preview	$2.50	-	$10.00
computer-use-preview	$3.00	-	$12.00
gpt-image-1	$5.00	$1.25	-
gpt-image-1-mini	$2.00	$0.20	-

"""

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

# ----------------- Utility / API wrapper -----------------

client = OpenAI()  # uses OPENAI_API_KEY env var

def _safe_get_usage(resp: Any) -> Dict[str, Optional[int]]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    prompt_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        try:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        except Exception:
            total_tokens = None
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}

def _looks_like_refusal(text: str) -> bool:
    """
    Heuristic to detect model refusals / safety responses.
    Returns True if the text contains common refusal phrases.
    """
    if not text:
        return True
    txt = text.lower()
    triggers = [
        "i'm sorry", "i am sorry", "i cannot", "i can't", "i cannot help", "i can't help",
        "i can't provide", "i cannot provide", "i'm not able to", "i am not able to",
        "can't assist", "cannot assist", "i won't", "i will not", "i must refuse",
        "i'm unable", "i am unable", "can't give", "cannot give", "i can't share",
        "i cannot share", "i'm sorry but", "i'm sorry, but"
    ]
    return any(t in txt for t in triggers)

# --------------------
# API call and retry
# --------------------
def call_responses_api(messages: List[Dict[str, str]],
                       model: str,
                       #temperature: float,   -> GPT-5 Responses API does not support temperature param any longer
                       #max_output_tokens: int,
                       timeout: int = 60) -> Tuple[str, Dict[str, Optional[int]], Any]:
    """
    Call OpenAI Responses API (synchronous).
    Returns (text, usage_dict, raw_response).
    May raise exceptions.
    """
    resp = client.responses.create(
        model=model,
        input=messages,
        #temperature=temperature,
        #max_output_tokens=max_output_tokens,
        timeout=timeout,
    )
    text = getattr(resp, "output_text", None)
    if text is None:
        # try structured extraction
        try:
            out = getattr(resp, "output", None)
            if out and len(out) > 0:
                first = out[0]
                content = getattr(first, "content", None)
                # content may be list of dicts or objects
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, dict):
                        text = first_content.get("text", "")
                    else:
                        text = getattr(first_content, "text", "") or ""
        except Exception:
            text = ""
    if text is None:
        text = ""
    usage = _safe_get_usage(resp)
    return text.strip(), usage, resp

def robust_call(messages: List[Dict[str, str]],
                model: str,
                #temperature: float,
                #max_output_tokens: int,
                max_retries: int = 5,
                base_backoff: float = 1.0,
                timeout: int = 60) -> Tuple[Optional[str], Dict[str, Optional[int]], Optional[str], Optional[Any]]:
    """
    Retry wrapper. Returns (text_or_None, usage_dict, error_message_or_None, raw_response_or_None).
    On failure after retries returns text=None and error_message with traceback.
    """
    attempt = 0
    while True:
        try:
            text, usage, raw = call_responses_api(messages, model, timeout=timeout)
            # if text empty, treat as refusal detected (we will mark it downstream)
            return text, usage, None, raw
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                tb = traceback.format_exc()
                return None, {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}, f"{type(e).__name__}: {str(e)} | traceback:{tb}", None
            backoff = base_backoff * (2 ** (attempt - 1))
            backoff = backoff * (0.8 + 0.4 * (time.time() % 1))
            print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}", file=sys.stderr)
            time.sleep(backoff)

# --------------------
# Row processing
# --------------------
def process_row(idx: int,
                row: Dict[str, str],
                system_prompt: Optional[str],
                model: str,
                #temperature: float,
                #max_output_tokens: int,
                max_retries: int,
                timeout: int) -> Dict[str, Any]:
    """
    Process a single prompt row.
    Returns a dict with output fields.
    """
    row_id = row.get("id", "")
    strategy = row.get("strategy", "")
    prompt_text = row.get("prompt", "")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    text, usage, error_msg, raw = robust_call(messages, model=model, max_retries=max_retries, timeout=timeout)

    # Determine status
    status = "ok"
    error_field = ""
    if error_msg is not None:
        status = "error"
        error_field = error_msg
        text_for_csv = ""
    else:
        # if text empty or heuristic refusal -> mark refused
        if _looks_like_refusal(text):
            status = "refused"
            error_field = "model_refusal_or_empty_response"
            text_for_csv = text or ""
        else:
            status = "ok"
            text_for_csv = text

    out = {
        "id": row_id,
        "strategy": strategy,
        "prompt": prompt_text,
        "response": text_for_csv,
        "status": status,
        "error": error_field,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "model_used": model,
        # raw_response is kept in memory only; not written to CSV to avoid huge cells
        "raw_response": raw,
    }
    return out

# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Batch Responses API CSV processor with rerun options")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path (id,strategy,prompt)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path to create/append")
    parser.add_argument("--system", "-s", required=True, help="Path to system prompt text file")
    parser.add_argument("--model", "-m", default="gpt-4o", help="Model name (default gpt-4o)")
    #parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--concurrency", "-c", type=int, default=3, help="Number of parallel workers (default 3)")
    #parser.add_argument("--max-output-tokens", type=int, default=1024, help="Max tokens for model output")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout (seconds)")
    parser.add_argument("--save-interval", type=int, default=5, help="Save progress every N completed rows")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")
    parser.add_argument("--price-input-per-1k", type=float, default=0.0, help="USD per 1000 input tokens (optional)")
    parser.add_argument("--price-output-per-1k", type=float, default=0.0, help="USD per 1000 output tokens (optional)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "continue", "rerun_failed", "rerun_ids"],
                        help="Mode: all | continue | rerun_failed | rerun_ids")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated IDs to rerun (used with rerun_ids)")
    args = parser.parse_args()

    # API key check
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # read system prompt
    if not os.path.isfile(args.system):
        print(f"ERROR: system prompt file not found: {args.system}", file=sys.stderr)
        sys.exit(1)
    with open(args.system, "r", encoding="utf-8") as fh:
        system_prompt = fh.read().strip()

    # read input CSV
    df = pd.read_csv(args.input, dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}
    if not {"id", "strategy", "prompt"}.issubset(set(cols_lower.keys())):
        print("ERROR: input CSV must contain columns: id, strategy, prompt", file=sys.stderr)
        print("Found columns:", list(df.columns), file=sys.stderr)
        sys.exit(1)
    df = df.rename(columns={cols_lower["id"]: "id", cols_lower["strategy"]: "strategy", cols_lower["prompt"]: "prompt"})
    rows = df.to_dict(orient="records")

    # prepare output
    output_columns = ["id", "strategy", "prompt", "response", "status", "error",
                      "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd", "model_used"]
    if os.path.exists(args.output) and not args.overwrite:
        out_df = pd.read_csv(args.output, dtype=str)
        for c in output_columns:
            if c not in out_df.columns:
                out_df[c] = ""
    else:
        out_df = pd.DataFrame(columns=output_columns)

    # determine which rows to process based on mode
    to_process = []  # list of (idx, row)
    if args.mode == "all":
        for idx, r in enumerate(rows):
            to_process.append((idx, r))
    elif args.mode == "continue":
        existing_ids = set(out_df["id"].astype(str).tolist())
        for idx, r in enumerate(rows):
            if str(r.get("id", "")) in existing_ids:
                print(f"[SKIP] id={r.get('id')} already present (continue mode).")
                continue
            to_process.append((idx, r))
    elif args.mode == "rerun_failed":
        # collect ids in output with status error or refused
        if out_df.empty:
            print("[INFO] Output file empty; no failed rows to rerun.")
        else:
            failed_mask = out_df["status"].isin(["error", "refused"]) if "status" in out_df.columns else [False]*len(out_df)
            failed_ids = set(out_df.loc[failed_mask, "id"].astype(str).tolist())
            if not failed_ids:
                print("[INFO] No failed/refused rows found in output CSV.")
            else:
                id_to_row = {str(r.get("id","")): (i, r) for i, r in enumerate(rows)}
                for fid in failed_ids:
                    if fid in id_to_row:
                        to_process.append(id_to_row[fid])
                    else:
                        print(f"[WARN] failed id {fid} not found in input prompts; skipping.")
    elif args.mode == "rerun_ids":
        if not args.ids:
            print("ERROR: --ids must be provided for rerun_ids mode (comma separated)", file=sys.stderr)
            sys.exit(1)
        requested = {s.strip() for s in args.ids.split(",") if s.strip()}
        id_to_row = {str(r.get("id","")): (i, r) for i, r in enumerate(rows)}
        for rid in requested:
            if rid in id_to_row:
                to_process.append(id_to_row[rid])
            else:
                print(f"[WARN] requested id {rid} not found in input CSV; skipping.")

    total_to_process = len(to_process)
    if total_to_process == 0:
        print("[INFO] Nothing to process based on the selected mode. Exiting.")
        return

    print(f"[INFO] Mode={args.mode} Will process {total_to_process} rows (concurrency={args.concurrency})")

    # bookkeeping
    results_by_index = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    completed_count = 0

    # run thread pool
    workers = max(1, int(args.concurrency))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {}
        for (idx, row) in to_process:
            future = ex.submit(process_row, idx, row, system_prompt, args.model, args.max_retries, args.timeout)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = {
                    "id": rows[idx].get("id", ""),
                    "strategy": rows[idx].get("strategy", ""),
                    "prompt": rows[idx].get("prompt", ""),
                    "response": "",
                    "status": "error",
                    "error": f"Unhandled worker exception: {type(e).__name__}: {str(e)}",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "model_used": args.model,
                    "raw_response": None,
                }

            # compute cost if token counts available
            pt = out.get("prompt_tokens")
            ct = out.get("completion_tokens")
            cost = 0.0
            if pt is not None and args.price_input_per_1k > 0.0:
                try:
                    cost += (float(pt) / 1000.0) * float(args.price_input_per_1k)
                except Exception:
                    pass
            if ct is not None and args.price_output_per_1k > 0.0:
                try:
                    cost += (float(ct) / 1000.0) * float(args.price_output_per_1k)
                except Exception:
                    pass
            out["cost_usd"] = round(cost, 8)

            # accumulate totals
            try:
                if pt is not None:
                    total_prompt_tokens += int(pt)
            except Exception:
                pass
            try:
                if ct is not None:
                    total_completion_tokens += int(ct)
            except Exception:
                pass
            total_cost += cost

            results_by_index[idx] = out
            completed_count += 1
            print(f"[DONE] processed index={idx} id={out.get('id')} status={out.get('status')} tokens={out.get('total_tokens')} cost=${out.get('cost_usd')}")

            # periodic save: append only rows not already in out_df (C-1 behavior)
            if completed_count % args.save_interval == 0:
                existing_ids = set(out_df["id"].astype(str).tolist())
                new_rows = []
                for i in sorted(results_by_index.keys()):
                    r = results_by_index[i]
                    if str(r.get("id","")) in existing_ids:
                        continue
                    new_rows.append({
                        "id": r.get("id", ""),
                        "strategy": r.get("strategy", ""),
                        "prompt": r.get("prompt", ""),
                        "response": r.get("response", ""),
                        "status": r.get("status", ""),
                        "error": r.get("error", ""),
                        "prompt_tokens": r.get("prompt_tokens"),
                        "completion_tokens": r.get("completion_tokens"),
                        "total_tokens": r.get("total_tokens"),
                        "cost_usd": r.get("cost_usd"),
                        "model_used": r.get("model_used"),
                    })
                if new_rows:
                    out_df = pd.concat([out_df, pd.DataFrame(new_rows)], ignore_index=True)
                    out_df.to_csv(args.output, index=False)
                    print(f"[INFO] Saved progress to {args.output} ({len(new_rows)} new rows appended)")

    # final append of remaining results
    existing_ids = set(out_df["id"].astype(str).tolist())
    final_new = []
    for i in sorted(results_by_index.keys()):
        r = results_by_index[i]
        if str(r.get("id","")) in existing_ids:
            continue
        final_new.append({
            "id": r.get("id", ""),
            "strategy": r.get("strategy", ""),
            "prompt": r.get("prompt", ""),
            "response": r.get("response", ""),
            "status": r.get("status", ""),
            "error": r.get("error", ""),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "total_tokens": r.get("total_tokens"),
            "cost_usd": r.get("cost_usd"),
            "model_used": r.get("model_used"),
        })
    if final_new:
        out_df = pd.concat([out_df, pd.DataFrame(final_new)], ignore_index=True)
    out_df.to_csv(args.output, index=False)

    # summary
    measured_total_tokens = total_prompt_tokens + total_completion_tokens
    print("\n=== SUMMARY ===")
    print(f"Processed rows this run: {len(results_by_index)}")
    print(f"Measured prompt tokens: {total_prompt_tokens}")
    print(f"Measured completion tokens: {total_completion_tokens}")
    print(f"Measured total tokens: {measured_total_tokens}")
    if args.price_input_per_1k > 0.0 or args.price_output_per_1k > 0.0:
        print(f"Estimated total cost (USD): ${total_cost:.8f}")
        if len(results_by_index) > 0:
            print(f"Average cost per processed row: ${ (total_cost / len(results_by_index)):.8f }")
    else:
        print("Cost not calculated (no per-1k prices provided).")
    print(f"Results appended to: {args.output}")
    print("Done.")

if __name__ == "__main__":
    main()