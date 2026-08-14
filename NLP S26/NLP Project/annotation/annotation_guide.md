# Annotation Guide — LongBench Diagnostic Error Taxonomy

**Version:** 1.0  
**Prepared by:** Rudrani  
**Task:** Label incorrect model predictions with a primary error category and a hallucination flag.

---

## 1. Overview

You will be shown instances where the model predicted the **wrong answer** on a long-context multiple-choice QA task. For each instance you will see:

- The context (truncated to 8K or 32K tokens)
- The question
- The four answer choices (A/B/C/D)
- The model's prediction
- The ground truth answer

Your job is to assign:
1. **One primary label** from: `RF`, `RsF`, `INC`
2. **One hallucination flag**: `HALL = 0` or `HALL = 1`

---

## 2. Decision Flowchart

Work through these steps **in order**. Stop at the first matching condition.

**Step 1.** Is the model's output a single letter A, B, C, or D?
- **No** → Label = **INC**. Stop.

**Step 2.** Is there any passage in the context that supports the predicted (wrong) answer?
- **No** → The model picked an answer with no contextual grounding. Label = **RF**. Continue to Step 4 for HALL.

**Step 3.** Is there a passage in the context that clearly supports the *correct* answer?
- **Yes** → The evidence was present but the model reasoned incorrectly. Label = **RsF**. Continue to Step 4 for HALL.
- **No** → The question may be unanswerable from the truncated context. Still label **RF** (the model should not have committed to a wrong answer). Continue to Step 4 for HALL.

**Step 4.** Does the model's prediction reference a specific fact, name, number, date, or claim that does not appear **anywhere** in the provided context?
- **Yes** → `HALL = 1`
- **No** → `HALL = 0`

---

## 3. Label Definitions

### RF — Retrieval Failure
The model selects an answer for which **no supporting evidence** exists anywhere in the context. The failure is locating (or failing to locate) the right passage before any reasoning even begins.

### RsF — Reasoning Failure
The relevant evidence **is present** in the context and a careful human reader would identify it, but the model draws the wrong inference from it. The passage was found; the conclusion was wrong.

### INC — Instruction Non-Compliance
The model's output is **not a valid answer letter** (A/B/C/D), or it violates an explicit constraint stated in the prompt (e.g., outputs two letters, outputs a full sentence instead of a letter, refuses to answer). Use INC regardless of whether the underlying content was right or wrong — the format failure is the primary issue.

### HALL — Hallucination (secondary flag)
Set `HALL = 1` when the model's prediction or the reasoning it implies depends on a **specific fact not present in the context**: a named entity, a number, a date, a relationship, or a claim you cannot locate in the provided text. This flag is orthogonal to the primary label — for example, an RF error can also be HALL=1 if the model appears to have invented the basis for its answer.

---

## 4. Worked Examples

### 4.1 Retrieval Failure (RF) Examples

**Example RF-1**

> **Context:** A 7,000-token passage about the history of the Roman Empire, covering military campaigns, political structures, and trade routes. No mention of coinage reforms.  
> **Question:** Which emperor introduced the aureus coin reform described in the passage?  
> **Choices:** A. Augustus  B. Nero  C. Diocletian  D. Constantine  
> **Model prediction:** B  
> **Ground truth:** C  
> **Label:** RF, HALL=0  
> **Justification:** No passage about coinage reform exists in the context; the model retrieved nothing and guessed.

---

**Example RF-2**

> **Context:** A contract document discussing payment terms, delivery schedules, and warranty clauses. The arbitration clause is entirely absent due to truncation.  
> **Question:** According to the contract, which city is specified as the arbitration venue?  
> **Choices:** A. New York  B. London  C. Singapore  D. Paris  
> **Model prediction:** A  
> **Ground truth:** C  
> **Label:** RF, HALL=0  
> **Justification:** The arbitration clause was truncated out; no venue appears anywhere in the provided context. The model could not have retrieved it.

---

**Example RF-3**

> **Context:** A scientific paper on CRISPR applications in agriculture. The results section discusses yield improvements in wheat and maize. No data on rice is present.  
> **Question:** By what percentage did rice yield improve in the experiment described?  
> **Choices:** A. 12%  B. 18%  C. 23%  D. 31%  
> **Model prediction:** C  
> **Ground truth:** B  
> **Label:** RF, HALL=1  
> **Justification:** Rice yield data is absent from the context entirely. The model not only failed to retrieve evidence but referenced a specific percentage (23%) that does not appear anywhere — hence HALL=1.

---

### 4.2 Reasoning Failure (RsF) Examples

**Example RsF-1**

> **Context:** "...the treaty was signed on 14 March 1879, granting Bolivia access to the Pacific corridor for a period of 25 years, after which all rights reverted to Chile..."  
> **Question:** For how many years did Bolivia retain Pacific access under the treaty?  
> **Choices:** A. 10 years  B. 15 years  C. 25 years  D. 50 years  
> **Model prediction:** D  
> **Ground truth:** C  
> **Label:** RsF, HALL=0  
> **Justification:** The passage explicitly states 25 years. The model had the right sentence but produced the wrong answer — a reasoning error, not a retrieval failure.

---

**Example RsF-2**

> **Context:** "...Revenue in Q3 was $4.2M, up from $3.1M in Q2, representing a 35.5% increase..."  
> **Question:** What was the percentage revenue growth from Q2 to Q3?  
> **Choices:** A. 15.2%  B. 26.0%  C. 35.5%  D. 41.0%  
> **Model prediction:** B  
> **Ground truth:** C  
> **Label:** RsF, HALL=0  
> **Justification:** The exact figure (35.5%) appears verbatim in the context. The model found the right paragraph but picked the wrong number.

---

**Example RsF-3**

> **Context:** A biography passage stating that the author published her first novel in 1991 and her award-winning second novel in 1998, which won the Booker Prize.  
> **Question:** Which novel won the Booker Prize?  
> **Choices:** A. Her debut novel  B. Her second novel  C. Her third novel  D. A co-authored novel  
> **Model prediction:** A  
> **Ground truth:** B  
> **Label:** RsF, HALL=0  
> **Justification:** The passage clearly attributes the Booker Prize to the second novel. The model misattributed it to the debut — a reasoning/reading error on evidence that was present.

---

### 4.3 Instruction Non-Compliance (INC) Examples

**Example INC-1**

> **Model output:** "Based on the context, the answer appears to be either B or C, but I cannot determine which definitively."  
> **Ground truth:** B  
> **Label:** INC, HALL=0  
> **Justification:** The model was asked to output a single letter. Outputting a sentence with two candidate answers violates the format constraint.

---

**Example INC-2**

> **Model output:** "C and D"  
> **Ground truth:** A  
> **Label:** INC, HALL=0  
> **Justification:** Two letters were output. The instruction was to answer with a single letter only.

---

**Example INC-3**

> **Model output:** "" *(empty string)*  
> **Ground truth:** D  
> **Label:** INC, HALL=0  
> **Justification:** No answer was produced at all, violating the task constraint to answer with A, B, C, or D.

---

## 5. Counter-Examples (Looks Like X, Actually Y)

### Counter-Example 1: Looks like RsF, actually RF

> **Context:** A long legal document where the relevant clause about liability caps was cut off mid-sentence due to truncation.  
> **Question:** What is the maximum liability cap stated in the agreement?  
> **Model prediction:** A ($500,000)  
> **Ground truth:** C ($2,000,000)  
> **Why it looks like RsF:** The model may have found *some* dollar figure in the document.  
> **Why it is actually RF:** The actual liability cap clause is absent from the context. There is no passage that supports *either* the predicted answer or the correct answer. No retrieval was possible → **RF**.

---

### Counter-Example 2: Looks like RF, actually RsF

> **Context:** A dense technical passage. The answer appears in a subordinate clause buried 6,000 tokens into the document.  
> **Question:** What voltage does the device operate at?  
> **Model prediction:** B (12V)  
> **Ground truth:** A (5V)  
> **Why it looks like RF:** The model seems to have ignored the passage entirely.  
> **Why it is actually RsF:** If you search carefully, the sentence "…operates at 5V DC input…" is present in the context. The evidence was there — the model failed to correctly use it → **RsF**.  
> **Rule of thumb:** Always search the full context before assigning RF. If the correct evidence is findable by a patient human reader, it is RsF.

---

### Counter-Example 3: Looks like INC, actually RsF

> **Model output:** "The answer is B."  
> **Ground truth:** C  
> **Why it looks like INC:** The model didn't output just "B" — it output a sentence.  
> **Why it is actually RsF (or RF):** The model *did* commit to a single answer letter (B). The sentence wrapper is acceptable — extract_choice() would parse this as "B". Apply the flowchart from Step 2 to determine RF vs RsF.  
> **Rule:** INC applies only when no valid single-letter answer can be extracted, not when the letter is embedded in a sentence.

---

## 6. Hallucination Flag Instructions

Set `HALL = 1` if the model's prediction implies reliance on a **specific fact not present anywhere in the provided context**. Examples of hallucinated specifics:

- A named person, place, organisation, or product not mentioned in the context
- A specific number, date, percentage, or measurement not found in the context
- A causal claim or relationship that is not stated or inferable from the context

**HALL is independent of the primary label.** You can have:
- RF + HALL=1: No evidence in context, and the model appears to have invented the basis for its answer
- RsF + HALL=0: Evidence was present, model just reasoned wrong
- INC + HALL=1: Non-compliant format AND the content references fabricated facts

**When in doubt about HALL**, default to `HALL = 0`. Only flag hallucination when you are confident the specific fact is absent from the context.

---

## 7. Adjudication Protocol

1. After independent annotation, disagreements are identified automatically by `src/iaa.py`.
2. For every disagreement, **both annotators** discuss the instance and write a 2-sentence resolution to `annotation/adjudication_log.md` in this format:

```
Instance ID: <id>
Annotator A label: <label>  Annotator B label: <label>
Resolved label: <label>
Resolution: <Sentence 1 explaining why one label was chosen.> <Sentence 2 noting what made this instance ambiguous.>
```

3. The resolved label is entered into the `resolved_label` column of `iaa_pilot.csv`.
4. If consensus cannot be reached, escalate to the full team for a majority vote.
5. All adjudication log entries must be completed before IAA statistics are reported.
