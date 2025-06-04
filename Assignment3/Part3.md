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

### Troubleshooting

While setting up this task, we encountered an issue while finetuning the model. Therefore we had to downgrade the versions of `unsloth` to `2025.5.7` and `unsloth-zoo` to `2025.5.8` using the following commands:

```bash
pip uninstall unsloth unsloth-zoo
pip install unsloth==2025.5.7 unsloth-zoo==2025.5.8
```

## Comparison

### The question "How many r's are in the word strawberry? Explain why!" was given to both the native gemma3:1b and the fine-tuned gemma3:1b model ten times.

The text "Explain why!" was added, because the reasoning was in every case "The word strawberry contains $x$ _r_'s"

#### Native gemma3:1b

**Answers:**

<SOLUTION>The word “strawberry” has two ‘r’s. Each ‘r’ represents a distinct letter.

<SOLUTION>The word “strawberry” has three ‘r’s.

<SOLUTION>The word “strawberry” has one ‘r’.

<SOLUTION>The word "strawberry" has five "s" characters.

<SOLUTION>The word “strawberry” contains one ‘r’.

<SOLUTION>The word "strawberry" has four "s" characters.

<SOLUTION>The letter “r” appears three times in the word “strawberry”.

<SOLUTION>The letter "r" appears three times in the word "strawberry".

<SOLUTION>There are two ‘r’s in the word "strawberry".

<SOLUTION>The word “strawberry” has two ‘r’s. Each ‘r’ is a distinct letter and is present in the word.

#### Finetuned gemma3:1b

**Answers:**

<SOLUTION>The word "strawberry" contains the letter "s" twice. Therefore, the answer is 2.

<SOLUTION>The word “strawberry” contains four “r”s. Each ‘r’ represents a distinct letter in the word.

<SOLUTION>The word “strawberry” contains four ‘r’s. Each letter in the word “strawberry” is pronounced ‘r’ as it is written.

<SOLUTION>The word “strawberry” contains one ‘r’. This is because the ‘r’ is the only letter that appears in the word.

<SOLUTION>The letter ‘r’ appears 5 times in the word “strawberry”.

<SOLUTION> The word “strawberry” contains four ‘r’s. Each ‘r’ represents a distinct letter.

<SOLUTION>The word “strawberry” has four ‘r’s. Each ‘r’ represents a distinct sound.

<SOLUTION>The word “strawberry” has four ‘r’s. This is because the letter ‘r’ appears four times in the word.

<SOLUTION>The word “strawberry” has three ‘r’s. Each letter ‘r’ in the word represents a distinct sound.

<SOLUTION>The word “strawberry” has four ‘r’s. Each ‘r’ represents a distinct letter.

#### Comparison

While the model, which is not finetuned, answers three times correctly, the finetuned model only provides one time the correct answer. Anyhow the result seems to be pretty random. Most of the time in the model, which is not finetuned, does not even give an explaination, though it was asked for one. This seems to be better with the finetuned model. The finetuned model does not give good reasons for why the word strawberry contains $x$ _r_'s, but at least it tries to explain.
