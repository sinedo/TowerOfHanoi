# GRPO Fine-tuning

## Fine-tuning and setting up the ollama model from .gguf

The gemma3:1b model was finetuned using the provided Colab notebook (https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb#scrollTo=5tkTF5Hmlhl-). After finetuning the model was saved as .gguf . In order to use it with ollama, a Modelfile had to be created. With `ollama show --modelfile gemma3:1b` the modelfile used for the native gemma3:1b could be used as a template. During finetuning, a system prompt was used:
```
"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
```

This system prompt was also included in the modelfile. The modelfile then looked like the following:
```
FROM ./gemma-3-finetune.Q8_0.gguf

TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""

PARAMETER stop <end_of_turn>
PARAMETER temperature 1
PARAMETER top_k 64
PARAMETER top_p 0.95

SYSTEM """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""
```

The ollama model was then created using `ollama create gemma3:1b-finetune -f /workspace/gemma-3-1b-finetune/Modelfile`.

## Comparison 

### The question "How many r's are in the word strawberry?" was given to both the native gemma3:1b and the fine-tuned gemma3:1b model.

#### Native gemma3:1b

**Answer:**

<SOLUTION>There are three ‘r’s in the word strawberry.

#### Native gemma3:1b-finetune

**Answer:**

<SOLUTION>There are three ‘r’s in the word strawberry.


### The same question was also asked using the provided python pseudo code interface for Ollama (Part2.1)**

#### Test gemma3:1b


**Answer:**
<think>The word “strawberry” has one ‘r’.

<ans>One


#### Test gemma3:1b-finetune

**Answer:**

<think>The word “strawberry” contains one “r”.
<ans>One</ans)