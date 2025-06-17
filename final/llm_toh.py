from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import time

def extract_solution_block(output: str) -> str:
    start = output.find("<solution>")
    end = output.find("</solution>")
    if start == -1 or end == -1:
        return None
    return output[start + len("<solution>"):end].strip()

MODEL_NAME = "gemma3:12b"
NUM_ATTEMPTS = 5

with open("/root/catkin_ws/src/om_position_controller/scripts/current_state.txt") as file:
    content = file.readlines()[-3:]

peg_A = [int(float(x)) for x in content[0].split(',') if float(x) != 0.0]
peg_B = [int(float(x)) for x in content[1].split(',') if float(x) != 0.0]
peg_C = [int(float(x)) for x in content[2].split(',') if float(x) != 0.0]

current_state_block = f"""<current_state> 
Current State:
A: {peg_A}
B: {peg_B}
C: {peg_C}
</current_state>
"""

print(current_state_block)

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
- Try to find a **valid sequence of moves** (not necessarily optimal).

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
Put only the final legal move sequence here, one move per line.
</solution>
""")

prompt_2_template = ChatPromptTemplate.from_template("""
You are a strict and logical assistant evaluating multiple candidate solutions to a stack-based puzzle.

## Stack Rules Recap:

- LIFO stacks (Last-In, First-Out)
- Each stack holds integers in **strict descending order** from bottom to top.
- You may only move the **top** element of a stack.
- You may only add an element to a stack if it is **smaller** than the current top element.
- Use move format: `MD[value][source][destination]` (e.g. MD3AB = move 3 from A to B).

## Goal:

From the possible solutions, identify the **shortest valid** one that ends with **[3, 2, 1]** in **one** stack (A, B, or C).  
Reject any sequence that:
- Violates move legality
- Doesn’t result in a stack with [3, 2, 1]
- Uses invalid moves or unknown stack IDs

Your output should include:

<thinking>
For each candidate:
- Validate each move's legality step-by-step
- Track current state of all stacks after each move
- Reject illegal or incomplete sequences
- Identify and keep the shortest valid one
</thinking>

<solution>
Only output the **first move** of the best valid sequence.
</solution>

{current_state}
{possible_solutions}
""")

llm = ChatOllama(model=MODEL_NAME)
parser = StrOutputParser()
solutions = []

# Phase 1: Generate possible solutions
for i in range(NUM_ATTEMPTS):
    chain = prompt_1_template | llm | parser
    result = chain.invoke({"current_state": current_state_block})
    sol = extract_solution_block(result)
    if sol:
        solutions.append(sol.strip())
        print(f"Solution {i+1}:\n{sol.strip()}\n")
    else:
        print(f"Solution {i+1}: No valid solution found.\n")
    time.sleep(2)

# Phase 2: Validate and pick best first move
if solutions:
    unique_solutions = list(set(solutions))
    formatted_solutions = "\n".join(
        f"<possible_solution>\n{sol}\n</possible_solution>" for sol in unique_solutions
    )
    final_chain = prompt_2_template | llm | parser
    final_response = final_chain.invoke({
        "current_state": current_state_block,
        "possible_solutions": formatted_solutions
    })

    print("Final Result:\n")
    print(final_response)
else:
    print("No valid solutions collected to evaluate.")
