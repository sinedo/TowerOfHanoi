import requests
import json
import sys
def generate_with_ollama(prompt, model="gemma3:4b", system="You are given a problem, and you need to think about it and provide an answer. Put your reasoning between <think></think> and your poposed answer between <ans></ans>."):
    url = "http://localhost:11434/api/generate"
    payload = {
    "model": model,
    "prompt": prompt,
    "system": system,
    "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    
# Example usage
if __name__ == "__main__":

    dict_reasoning_problems = {}
    dict_reasoning_problems["Simple deductive reasoning"] = "All dogs are mammals. Fido is a dog. Is Fido a mammal?"
    dict_reasoning_problems["Multi-Step Deductive Reasoning / Basic Inference"] = "If the library is open, Sarah will go there. If Sarah goes to the library, she will borrow a history book. The library is open. Will Sarah borrow a history book?"
    dict_reasoning_problems["Logic Puzzle with Constraints"] = "There are three friends: Alex, Ben, and Chris. One of them is a doctor, one is a lawyer, and one is an engineer. Alex is not the lawyer. Ben is not the doctor. The engineer is younger than Chris. Ben is older than the lawyer. Who is the doctor, who is the lawyer, and who is the engineer?"
    dict_reasoning_problems["Reasoning with Ambiguity and Abduction"] = "You enter a room and find a shattered vase on the floor, an open window with a muddy paw print on the sill, and your cat looking unusually guilty nearby. Your dog, who loves to chase squirrels, is nowhere to be seen, but you remember letting him out into the garden (which is accessible from the window) about 10 minutes ago. What are the most plausible explanations for the broken vase? Rank them from most to least likely, and explain what additional information could help you confirm or deny each hypothesis."
    dict_reasoning_problems["Complex Counterfactual and Systems Reasoning"] = "Imagine a world where, starting in the year 1900, the average global human lifespan instantly doubled, but global birth rates remained exactly the same as they were in our actual 1900. All other historical events and technological developments up to 1900 remain unchanged. Describe three major, distinct, and significant global consequences (social, economic, environmental, or political) that would likely be observable by the year 1950 in this alternate reality."

    
    with open("Part2_2.md", "w") as f:
        for category in dict_reasoning_problems:

            prompt = dict_reasoning_problems[category]
            f.write(f"## Category: {category}\n\n")
            f.write(f"### Prompt: {prompt}\n\n")
            response = generate_with_ollama(prompt)
            f.write(f"### Response from default gemma3:4b model:\n{response}\n\n")
    
    
    with open("Part2_3.md", "w") as f:
        prompt = "How many r's are in the word strawberry?"

        
        f.write(f"## Test gemma3:1b\n\n")
        f.write(f"### Prompt: {prompt}\n\n")
        response = generate_with_ollama(prompt, model="gemma3:1b")
        f.write(f"### Response from default gemma3:1b model:\n{response}\n\n")

        f.write(f"## Test gemma3:4b\n\n")
        f.write(f"### Prompt: {prompt}\n\n")
        response = generate_with_ollama(prompt)
        f.write(f"### Response from default gemma3:4b model:\n{response}\n\n")
    """
    prompt = "How many r's are in the word strawberry?"
    print(f"## Test gemma3:1b\n\n")
    print(f"### Prompt: {prompt}\n\n")
    response = generate_with_ollama(prompt, model="gemma3:1b")
    print(f"### Response from default gemma3:1b model:\n{response}\n\n")

    print(f"## Test gemma3:1b-finetune\n\n")
    print(f"### Prompt: {prompt}\n\n")
    response = generate_with_ollama(prompt, model="gemma3:1b-finetune")
    print(f"### Response from gemma3:1b-finetune model:\n{response}\n\n")
    """