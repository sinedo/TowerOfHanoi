from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import time

def extract_solution_block(output: str) -> str:
    start = output.find("<solution>")
    end = output.find("</solution>")
    if start == -1 or end == -1:
        return "NO SOLUTION TAGS FOUND"
    return output[start + len("<solution>"):end].strip()

MODELS = ["qwen3:4b",
          "qwen3:8b",
          "gemma3:4b",
          # "deepseek-r1:8b",
          "gemma3:12b"]
ATTEMPTS = [1, 3, 5]


with open("current_state.txt") as file:
    content = file.readlines()[-3:]

peg_A = np.array([int(float(x)) for x in content[0].split(',') if float(x) != 0.0])
peg_B = np.array([int(float(x)) for x in content[1].split(',') if float(x) != 0.0])
peg_C = np.array([int(float(x)) for x in content[2].split(',') if float(x) != 0.0])

current_state_block = f"""<current_state>
Current State:
A: {peg_A}
B: {peg_B}
C: {peg_C}
</current_state>
"""

print(f"A: {peg_A}")
print(f"B: {peg_B}")
print(f"C: {peg_C}")


prompt_1_template = ChatPromptTemplate.from_template("""
You are a helpful assistant simulating stack-based logic with strict constraints.

## Problem Setup:

- There are 3 stacks: A, B, and C.
- Each stack behaves like a **LIFO (Last In, First Out)** structure.
- Stack format: [bottom, ..., top] — i.e., the **rightmost element is on top**.

## Stack Rules:

- You may only move the **top element** of a stack.
- Elements can **only be added to the top** of a stack.
- **Descending constraint**: Elements in a stack must strictly **decrease** from bottom to top.
    - i.e., **you may only place a smaller number on top of a larger one**.
- All values are integers; higher numbers = "larger".

## Move Syntax:

- Use format: `MD[value][source][dest]`
- Example: `MD3AB` = move value 3 from stack A to stack B.

## Objective:

- Move elements such that **one** of the stacks (A, B, or C) ends up holding **[3, 2, 1]** (in that order — 3 on bottom, 1 on top).
- All moves must be legal and obey the rules above.
- Try to find a **valid sequence of moves**

{current_state}

## Task:

1. Simulate the stacks after each move to ensure legality.
2. Reject any illegal sequences (e.g., wrong order, invalid destination).
3. Stop once one stack has all 3 elements in correct descending order.

## Output Format:

<thinking>
- For each move, explain:
- What is being moved.
- Why the move is legal (or not).
- The resulting state of all stacks.
</thinking>

<solution>
ALWAYS INCLUDE THE SOLUTION TAGS <solution> </solution> and never change them
Put only the final legal move sequence here between the solution tags.
seperate the moves by linebreaks
</solution>
""")

prompt_2_template = ChatPromptTemplate.from_template("""
You are a strict and logical assistant evaluating multiple candidate solutions to a stack-based puzzle.

## Stack Rules Recap:

- LIFO stacks (Last-In, First-Out)
- Each stack holds integers in **strict descending order** from bottom to top.
- You may only move the **top** element of a stack.
- You may only add an element to a stack if it is **smaller** than the current top element.
- Use move format: `MD[value][source][destination]` (e.g. MD2BC = move 2 from B to C).

## Goal:

From the possible solutions, identify the **shortest valid** one that ends with **[3, 2, 1]** in **one** stack (A, B, or C).
Reject any sequence that:
- Violates move legality
- Doesn’t result in a stack with [3, 2, 1]
- Uses invalid moves or unknown stack IDs
- Check if the order of the moves makes sense MD1AB or MD1BA, switch if necessery

Your output should include:

<thinking>
For each candidate:
- Validate each move's legality step-by-step
- Track current state of all stacks after each move
- Reject illegal or incomplete sequences
- Identify and keep the shortest valid one
</thinking>

<solution>
ALWAYS INCLUDE THE SOLUTION TAGS <solution> </solution> and never change them
Only output the **first move** of the best valid sequence.

</solution>

{current_state}
{possible_solutions}
""")


parser = StrOutputParser()

with open("results.md", "w") as results_file:
    results_file.write("# Tower of Hanoi Model Stats\n\n")
    results_file.write("\n")
    results_file.write(f"A: {peg_A}")
    results_file.write(f"B: {peg_B}")
    results_file.write(f"C: {peg_C}")
    results_file.write("\n")


    for model_name in MODELS:
        for attempt_count in ATTEMPTS:
            print(f"Running model: {model_name}, attempts: {attempt_count}")
            llm = ChatOllama(model=model_name)
            solutions = []

            # Phase 1: Generate solutions
            for i in range(attempt_count):
                chain = prompt_1_template | llm | parser
                result = chain.invoke({"current_state": current_state_block})
                sol = extract_solution_block(result)
                if sol:
                    solutions.append(sol)
                time.sleep(2)

            # Save solutions
            results_file.write(f"## Model: {model_name} | Attempts: {attempt_count}\n")
            for idx, sol in enumerate(solutions):
                results_file.write(f"**Attempt {idx + 1}:**\n\n{sol}\n\n")

            # Phase 2: Evaluate
            if solutions:
                unique_solutions = list(set(solutions))
                formatted_solutions = "\n".join(
                    f"<possible_solution>\n{sol}\n</possible_solution>" for sol in unique_solutions
                )
                eval_chain = prompt_2_template | llm | parser
                eval_response = eval_chain.invoke({
                    "current_state": current_state_block,
                    "possible_solutions": formatted_solutions
                })
                final_sol = extract_solution_block(eval_response)
                results_file.write(f"**Best First Move:** `{final_sol}`\n\n")
            else:
                results_file.write("**No valid solutions found.**\n\n")
            results_file.flush()
