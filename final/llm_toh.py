from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import time

def extract_solution_block(output: str) -> str:
    start = output.find("<solution>")
    end = output.find("</solution>")
    if start == -1 or end == -1:    # index not found
        return None
    return output[start + len("<solution>"):end].strip()



MODEL_NAME = "gemma3:12b"
NUM_ATTEMPTS = 10 # for first prompt


current_state_block = """<current_state> 
Current State:
A: [1]
B: []
C: [3,2]
</current_state>

"""

prompt_1_template = ChatPromptTemplate.from_template("""
You are a helpful assistant working with LIFO (Last In, First Out) stacks. 
These are special stacks with a unique constraint: 
they can only be filled in strict DESCENDING order, 
meaning smaller numbers/elements must always be on top of larger ones (except during initialization, but not by us). 

Setup:
- 3 stacks: A, B, C
- Stack format: [bottom element, x, x, x, top element] - rightmost element is the top
- Stack elements must increase strictly from bottom to top.
- Only the top element of a stack can be moved, and elements can only be added to the top
- Goal: Fill one of the stacks in descending order with all elements

Stack Rules:
- FILO behavior: only the top (rightmost) element can be moved
- DESCENDING constraint: once a stack has elements, any new element added to the top must be greater than the current top
- Elements can only be moved one at a time
- Use format: MD[value][source_stack][destination_stack]
- to check if a move is valid see if current top element is bigger then what you would place on it
Example: MD3AB means move value 3 from stack A to stack B
{current_state}
Goal State:
A or B or C: [3, 2, 1]
Task:
1. Analyze the current configuration
2. Verify the move is legal: Is the element plcaed on top? Can it legally go on the destination stack?
3. Verify that all elements are in one stack

Output Format:
- Put your step-by-step reasoning and validation in <thinking></thinking> tags
- Put only the final move sequence in <solution></solution> tags
- Each move should be on a separate line within the solution tags
Before providing your answer, verify the moves are legal and the moves are respect the descending constraint.
""")

prompt_2_template = ChatPromptTemplate.from_template("""
You are a helpful assistant working with LIFO (Last In, First Out) stacks. 
These are special stacks with a unique constraint: 
they can only be filled in strict DESCENDING order, 
meaning smaller numbers/elements must always be on top of larger ones (except during initialization, but not by us). 

Setup:
- 3 stacks: A, B, C
- Stack format: [bottom element, x, x, x, top element] - rightmost element is the top
- Stack elements must increase strictly from bottom to top.
- Only the top element of a stack can be moved, and elements can only be added to the top
- Goal: Fill one of the stacks in descending order with all elements

Stack Rules:
- FILO behavior: only the top (rightmost) element can be moved
- DESCENDING constraint: once a stack has elements, any new element added to the top must be greater than the current top
- Elements can only be moved one at a time
- Use format: MD[value][source_stack][destination_stack]
- to check if a move is valid see if current top element is bigger then what you would place on it
Example: MD3AB means move value 3 from stack A to stack B
{current_state}
Goal State:
A or B or C: [3, 2, 1]
Task:
1. Analyze the current configuration
2. Verify the move is legal: Is the element plcaed on top? Can it legally go on the destination stack?
3. Verify that all elements are in one stack
4. check the possible solution and return the correct one! if none is correct say so.

{possible_solutions}

Output Format:
- Put your step-by-step reasoning and validation in <thinking></thinking> tags
- Put only the final move sequence in <solution></solution> tags
Before providing your answer, verify the moves are legal and the moves are respect the descending constraint. check also if some moves in the sequence can ommited
""")

llm = ChatOllama(model=MODEL_NAME)
parser = StrOutputParser()

solutions = []

for i in range(NUM_ATTEMPTS):
    chain = prompt_1_template | llm | parser
    result = chain.invoke({"current_state": current_state_block})
    sol = extract_solution_block(result)
    if sol:
        solutions.append(sol)
        print(f"Solution {i+1}:\n{sol}\n")
    else:
        print(f"Solution {i+1}: No valid solution found.\n")
    time.sleep(1)  # Prevent flooding Ollama


if solutions:
    formatted_solutions = "\n".join(
        f"<possible_solution>\n{sol}\n</possible_solution>" for sol in solutions
    )
    final_chain = prompt_2_template | llm | parser
    final_response = final_chain.invoke({
        "current_state": current_state_block,
        "possible_solutions": formatted_solutions
    })

    print("Final Result:\n")
    print(final_response)
else:
    print("No solutions collected to validate.")