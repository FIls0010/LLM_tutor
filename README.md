```
python3 llm_batch_processor.py \
        --input prompts.csv \
        --output responses.csv \
        --system system_prompt.txt \
        --model gpt-5 \
        --concurrency 3 \
        --save-interval 5 \
        --price-input-per-1k 0.0025 \
        --price-output-per-1k 0.01
```

```
python3 llm_evaluator.py \
        --input responses.csv \
        --output evaluated_responses.csv \
        --judge-model gpt-5 \
        --rubric rubric.txt \
        --concurrency 3 \
        --price-input-per-1k 0.0025 \
        --price-output-per-1k 0.01
```

