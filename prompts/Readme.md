# Zero-Shot vs One-Shot vs Few-Shot Prompting (LLM Notes)

## 1. Zero-Shot Prompting

Definition:
The model is given only an instruction, no examples.

How it works:
The LLM uses its pre-trained knowledge to understand the task and respond.

Example:
Prompt:
Classify the sentiment of the sentence:
"I waited 2 hours and the food was cold."

Output:
Negative

Pros:

* Fast
* Cheap (small token usage)
* Works well for simple tasks

Cons:

* Low accuracy for structured tasks
* Format may be inconsistent

Best Use Cases:

* Chatbots
* General Q&A
* Summaries
* Brainstorming

---

## 2. One-Shot Prompting

Definition:
The model is given exactly one example before the real question.

How it works:
The model learns the pattern from a single demonstration.

Example:
Prompt:
Classify sentiment.

Text: "I love this phone"
Sentiment: Positive

Text: "Battery drains in 1 hour"
Sentiment:

Output:
Negative

Pros:

* Better reliability than zero-shot
* Helps enforce format

Cons:

* Still inconsistent for complex logic

Best Use Cases:

* Small formatting tasks
* Label prediction
* API responses

---

## 3. Few-Shot Prompting

Definition:
The model is given multiple examples (typically 3–10) before the actual input.

How it works:
The model performs in-context learning and infers rules from patterns.

Example:
Prompt:
Convert English to SQL.

English: Show all users
SQL: SELECT * FROM users;

English: Get active users
SQL: SELECT * FROM users WHERE status='active';

English: Get users older than 18
SQL:

Output:
SELECT * FROM users WHERE age > 18;

Pros:

* High accuracy
* Consistent structured output
* Ideal for automation pipelines

Cons:

* More tokens → higher cost & latency

Best Use Cases:

* Data extraction
* JSON generation
* Code generation
* Production LLM systems

---

## Key Differences

Zero-Shot → No examples → Model guesses
One-Shot → One example → Model imitates
Few-Shot → Multiple examples → Model learns pattern

---

## Quick Comparison Table

Feature: Accuracy
Zero-Shot: Low
One-Shot: Medium
Few-Shot: High

Feature: Cost
Zero-Shot: Cheapest
One-Shot: Low
Few-Shot: Highest

Feature: Consistency
Zero-Shot: Poor
One-Shot: Good
Few-Shot: Very Good

Feature: Typical Usage
Zero-Shot: Chat & Q/A
One-Shot: Small formatting tasks
Few-Shot: Production pipelines

---

## Interview One-Liner

Zero-shot uses model knowledge,
One-shot teaches format,
Few-shot teaches behavior.
