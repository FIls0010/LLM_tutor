# LLM Behaviour Design & Validation Pipeline

> *(Originally developed and validated as an AI Tutor testing pipeline)*

A **general-purpose toolkit for designing, stress-testing, and validating the behaviour of LLM-based chatbots** — without writing code.

---

## What Is This?

This repository provides a **behaviour design and evaluation pipeline** for Large Language Model (LLM) chatbots.

You define how a chatbot should behave using a **system prompt**, test that behaviour against many inputs at scale, and optionally **automatically evaluate** whether the chatbot followed your rules — all using simple text files and CSVs.

> **If you can edit a text file, you can use this system.**

---

## Primary Validated Use Case: AI Tutors (Academic Integrity)

This pipeline was originally built to design and validate **AI tutors for university courses**, with a strong focus on **academic integrity**.

The validated tutor behaviour:
- Helps students learn through explanation and guidance
- Encourages critical thinking
- **Refuses to give direct answers** to questions that contribute to grades or marks
- Detects graded-style questions and red-flag phrasing
- Resists jailbreaks, manipulation, and authority appeals
- Maintains a professional, supportive teaching tone

This AI tutor use case has been:
- Tested against **140+ adversarial prompts**
- Manually reviewed
- Automatically evaluated with a custom rubric
- **Demonstrated to produce zero direct answer leaks**

---

## Broader Applications (Not Limited to Tutoring)

Although tutoring is the flagship example, this pipeline is **not tutor-specific**.

It can be used to design, test, and validate **any constrained or safety-critical LLM behaviour**, including:

- Corporate compliance assistants  
- Mental-health or wellbeing chatbots with strict safety rules  
- Customer-support bots that must avoid legal or medical advice  
- Study coaches that guide without solving  
- Policy-restricted or refusal-heavy assistants  
- Comparative testing of different system prompt designs  

If you need confidence that a chatbot **consistently follows rules**, this pipeline is applicable.

---

## What This Repo Is (and Isn’t)

### ✅ This repo **does**
- Shape chatbot behaviour via system prompts
- Stress-test behaviour using CSV-based batch inputs
- Record refusals, errors, token usage, and cost
- Automatically evaluate behaviour using LLM-as-a-judge
- Support iterative refinement and validation

### ❌ This repo **does not**
- Provide a chatbot UI or frontend
- Replace your LMS or chatbot platform
- Train or fine-tune models
- Generate direct answers to graded questions (for tutor use cases)

Think of this as a **behaviour design, testing, and validation layer** — not the chatbot itself.

---

## 📚 Documentation (Start Here)

All detailed documentation lives in the **Wiki**.

👉 **[🏠 Home – Overview & Concepts](../../wiki/Home)**  
*(Recommended starting point)*

### Recommended Reading Order
1. **[🚀 Getting Started](../../wiki/Getting-Started)**  
   Installation, setup, and first run

2. **[📝 System Prompt Guide](../../wiki/System-Prompt-Guide)**  
   How behaviour is defined and safely customised

3. **[⚙️ Batch Processing](../../wiki/Batch-Processing)**  
   Testing chatbot behaviour at scale

4. **[📊 Automated Evaluation](../../wiki/Automated-Evaluation)**  
   LLM-as-a-judge scoring and analysis

5. **[🎓 Customization Examples](../../wiki/Customization-Examples)**  
   Tutor, CS, Biology, Humanities, Medical, and more

6. **[💰 Cost & Pricing](../../wiki/Cost-and-Pricing)**  
   Model selection and budget optimisation

7. **[🔧 Troubleshooting](../../wiki/Troubleshooting)**  
   Common setup and runtime issues

---

## Quick Start (Minimal)

```bash
python3 llm_batch_processor.py \
  --input prompts.csv \
  --output responses.csv \
  --system system_prompt.txt
```

Full setup and explanation:
**[🚀 Getting Started](../../wiki/Getting-Started)**  

---

## Who Is This For?

- Educators and course coordinators

- Academic researchers studying AI behaviour

- Educational technologists

- Safety and alignment researchers

- Anyone designing rule-bound or refusal-aware LLM systems

**No programming background required.**

---
##Licence
MIT Licence - free to use, modify, and distribute
---

##Final Note
This project exists to make responsible, constrained, and verifiable AI behaviour practical — not theoretical.

Whether you’re building an AI tutor or any other carefully bounded chatbot, this pipeline gives you control, visibility, and confidence.

👉 Start here: **[🏠 Home – Overview & Concepts](../../wiki/Home)**  
