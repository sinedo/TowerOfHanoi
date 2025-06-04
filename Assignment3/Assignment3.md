# Advanced Methods in Robot Learning Group 4 Report - Assignment 3

Dominik Adlberger
Edonis Berisha
Nikola Szucsich

## Part 1: Local LLM Setup with Ollama

1. Install Ollama on the system:
2. Run ollama run gemma3:4b in your terminal, Test the model with basic reasoning prompts and Document the model's responses and capabilities.
3. Explore Ollama functionality: Document Ollama's key features and limitations

___

To test the models functionality, basic reasoning questions were prompted. The input prompts are denoten by "Q:" with the answers right below.

### Basic reasoning prompts (gemma3:4b)

### Q: Explain the concept of recursion

Okay, let's break down the concept of recursion. It’s a powerful and often mind-bending idea in computer science, but once you grasp the core, it becomes quite intuitive.

**What is Recursion?**

Recursion is a programming technique where a function *calls itself* within its own definition. Think of it like a set of Russian nesting dolls (Matryoshiki dolls). Each doll contains a smaller version of itself until you get to the smallest, indivisible doll.

**Here's a breakdown of the key elements:**

1. **Base Case:** Every recursive function *must* have a base case. This is the condition that stops the recursion. Without it, the function would call itself infinitely, leading to a stack overflow error. The base case provides a direct answer, without needing further recursion.

2. **Recursive Step:** This is where the function calls itself, but with a *modified* input. The modification is designed to bring the input closer to the base case. Each call reduces the problem size.

**Example: Calculating Factorial (a Classic Recursion Illustration)**

Let's illustrate with the factorial function (n! = n *(n-1)* (n-2) *...* 1).

```python

def factorial(n):

"""

Calculates the factorial of a non-negative integer.

"""

if n == 0: # Base Case: Factorial of 0 is 1

return 1

else:

return n * factorial(n-1) # Recursive Step: Call itself with n-1



# Example usage:

result = factorial(5)

print(result) # Output: 120

```

**How it works step-by-step for `factorial(5)`:**

1. `factorial(5)`: `n` is 5. It's not 0, so it executes the `else` block. It returns `5 * factorial(4)`. It needs to calculate `factorial(4)`.

2. `factorial(4)`: `n` is 4. It returns `4 * factorial(3)`.

3. `factorial(3)`: `n` is 3. It returns `3 * factorial(2)`.

4. `factorial(2)`: `n` is 2. It returns `2 * factorial(1)`.

5. `factorial(1)`: `n` is 1. It returns `1 * factorial(0)`.

6. `factorial(0)`: `n` is 0. This hits the **base case**! It returns `1`.

Now the values are returned up the chain:

* `factorial(1)` returns `1 * 1 = 1`

* `factorial(2)` returns `2 * 1 = 2`

* `factorial(3)` returns `3 * 2 = 6`

* `factorial(4)` returns `4 * 6 = 24`

* `factorial(5)` returns `5 * 24 = 120`

**Why Use Recursion?**

* **Elegance and Readability:** For certain problems (especially those naturally defined recursively), recursion can lead to more concise and readable code.

* **Problem Decomposition:** Recursion provides a natural way to break down complex problems into smaller, self-similar subproblems.

* **Tree and Graph Traversal:** Recursion is very common when dealing with data structures like trees and graphs.

**Important Considerations:**

* **Stack Overflow:** Each recursive call adds a new frame to the call stack. If the recursion goes too deep (e.g., not enough base cases or problems are too complex), you can get a stack overflow error.

* **Performance:** Recursion can sometimes be less efficient than iterative (loop-based) solutions due to the overhead of functioncalls. However, for some problems, it’s the most natural and performant approach.

**Other Common Recursion Examples:**

* **Fibonacci Sequence:** Calculating the Fibonacci numbers is frequently done recursively.

* **Tree Traversal:** Walking through the branches of a tree.

* **Sorting Algorithms:** Some sorting algorithms (like quicksort and mergesort) are inherently recursive.

**Resources for Further Learning:**

* **Khan Academy - Recursion:**

[https://www.khanacademy.org/computing/computer-science/algorithms/recursion](<https://www.khanacademy.org/computing/computer-science/[https://www.khanacademy.org/computing/computer-science/algorithms/recursion>](<https://www.khanacademyorg/computing/computer-science/algorithms/recursion>)

* **GeeksforGeeks - Recursion:**

[https://www.geeksforgeeks.org/recursion-in-python/](https://www.geeksforgeeks.org/recursion-in-python/)

Do you want me to:

* Explain recursion using a different example?

* Compare recursion to iteration (loops)?

* Discuss the potential pitfalls of recursion?

### Q: All birds have feathers. A robin is a bird. Does a robin have feathers?

Yes, a robin has feathers.

This is a classic example of a deductive argument, and it’s a valid one! Here’s the breakdown:

* **Premise 1:** All birds have feathers. (This is a general rule)

* **Premise 2:** A robin is a bird. (This places the robin within the category of ‘bird’)

* **Conclusion:** Therefore, a robin has feathers.

Because the conclusion follows logically from the given premises, the answer is definitively yes.

Would you like to try another logic puzzle, or perhaps explore a different aspect of recursion?

### Q: The last three times it was cloudy like this, it rained within an hour. It's cloudy like this now. What might happen soon?

This is a great example of a *Barnum Effect* (also known as the Forer Effect) combined with a bit of probabilistic thinking. Here’s a breakdown of what might happen, and why the situation feels so likely:

**What's likely to happen (and why):**

It’s *likely* that it will rain soon. The initial observations – “the last three times it was cloudy like this, it rained within an hour” – have created a strong *perception* of a pattern. Your brain is trying to find connections, even if those connections aren't truly statistically significant.

**Here’s what’s going on psychologically:**

* **Confirmation Bias:** You’re more likely to notice and remember instances where it rained after cloudy periods, reinforcing the idea that there’s a connection.

* **The Barnum Effect:** The description ("cloudy like this") is vague enough to apply to a wide range of weather conditions. This makes it seem like a more reliable predictor than it actually is.

* **Small Sample Size:** Three instances are a very small sample size. Random chance alone could account for a higher-than-expected probability of rain occurring after a cloudy period.

**What *might* happen:**

* **It could rain:** There's a higher chance than a completely random occurrence, but it’s still not guaranteed. The weather is influenced by complex atmospheric patterns.

* **It might clear up:** The clouds could dissipate.

* **It could rain a little, then clear:** A brief shower followed by sunshine is a common weather pattern.

**Important Note:** Weather is notoriously unpredictable over short periods. While past observations can influence your perception, they don’t guarantee future outcomes.

**Do you want to talk about:**

* How statistical significance works?

* Other examples of patterns being perceived when they don't exist?

* A different logical puzzle?

### Q: Your friend, who is usually punctual, is an hour late for your meeting. What are the most likely reasons for this?

Okay, this is a classic scenario for considering probabilities! Here’s a breakdown of the most likely reasons your punctual friend is an hour late, ranked in order of probability (based on common experience):

1. **Traffic Delay (60%):** This is overwhelmingly the most likely reason. Unexpected traffic jams, accidents, or significant delays on their route are incredibly common.

2. **Unexpected Personal Issue (25%):** Something unexpected happened that required their attention – a family emergency, a sudden illness, a flat tire, needing to help someone, etc. These are common and often difficult to share immediately.

3. **Misjudgment of Time (10%):** Even a punctual person can overestimate how long something will take, especially if they were distracted or engaged in a task before the meeting.

4. **Significant Delay with Transportation (5%):** A major delay with public transport (train, bus) or a car issue (breakdown, needing assistance) could be the culprit.

**Why this ranking?**

* Traffic is a consistently pervasive issue.

* Personal emergencies, while disruptive, aren’t *everyday* occurrences.

* Misjudging time is a relatively frequent human error.

**What *wouldn’t* be the most likely reasons?**

* Intentional rudeness (unlikely if they’re usually punctual)

* A complete inability to tell the time (very unlikely for someone generally considered punctual)

**To help me tailor the response further, would you like to:**

* Consider the context of your friend’s usual behavior? (e.g., do they tend to be overly cautious or prone to getting lost?)

* Discuss the role of assumptions in this scenario?

### Q: If 'up' is to 'down' as 'hot' is to '____', what is the missing word and why?

The missing word is **cold**.

Here’s the reasoning:

The relationship between "up" and "down" is one of **opposites** – they are diametrically opposed directions. “Hot” and “cold” are
also fundamentally opposite concepts – one represents high temperature, the other low.It’s a classic analogy that relies on recognizing opposing pairs. Would you like to try another analogy puzzle, or perhaps explore a different area of logic?

### Q: If you are facing North and turn 90 degrees clockwise, which direction are you now facing?

You are now facing **East**.

Turning 90 degrees clockwise from North takes you around to the East.

Do you want to try another directional puzzle, or would you like a different kind of challenge (e.g., logic puzzle, word problem)?

### Q: If 'yesterday's tomorrow' was Saturday, what day is 'tomorrow's yesterday'?

This is a clever wordplay puzzle! The answer is **Friday**.

Here’s the breakdown:

* “Tomorrow’s yesterday” refers to the *previous* day – the day before today.

* If ‘yesterday’s tomorrow’ (the next day) was Saturday, then today is Friday.

* Therefore, ‘tomorrow’s yesterday’ is the day *before* tomorrow, which is Friday.

It relies on understanding the relative positions of "tomorrow" and "yesterday" in relation to the present.

Would you like to attempt another word puzzle, or would you like a different type of challenge?

### Q: If pressing a switch turns on a light, and the light is not on, what can you infer about the switch (assuming the bulb is working and there's power)?

You can infer that the switch is **not making proper contact**.

Here’s the breakdown of why:

* **We’ve established:** The bulb is working, and there’s power to the circuit.

* **Therefore:** If the switch were functioning correctly, it would complete the circuit, allowing electricity to flow and thelight to turn on. Since the light isn’t on, the problem lies in the switch’s ability to reliably close the circuit.

Possible reasons for the switch failing to make proper contact include:

* It’s dirty or corroded.

* It’s mechanically damaged.

* It’s not fully engaged.

Do you want to explore a different logical deduction problem, or perhaps a puzzle involving a different scenario?

##### **Correction** Q: What about the switch being in the 'off' state?

You are absolutely correct! My apologies for the oversight. If the switch is in the ‘off’ state, then the logical inference is that the light is **off**.

It was a crucial and easily missed detail. Thanks for pointing it out – it’s a great example of how even seemingly straightforward scenarios can have nuances.

Do you want to try another logical deduction puzzle, or perhaps something different entirely?

### Q: You are in a room with two doors, Door A and Door B. One door leads to freedom, and the other leads to a trap. In the room are two guards. One guard always tells the truth, and the other guard always lies. You don't know which guard is which, and you don't know which door leads to freedom. You can ask only one guard one yes/no question. What question do you ask to guarantee you can find the door to freedom?

This is a classic logic puzzle! The question to ask is:

“If I were to ask the other guard which door leads to freedom, what would he say?”

**Here’s why this works:**

* **If you ask the truthful guard:** He’ll truthfully tell you what the liar would say – a lie about the door to freedom.

* **If you ask the liar:** He’ll lie about what the truthful guard would say. Therefore, he’ll also point you to the trap door.

In either case, the answer you receive will point you to the *wrong* door – the one that leads to the trap. Simply choose the *other* door.

It’s a clever trick that relies on understanding the inherent contradictions in the statements of the two guards.

Do you want to try another logic puzzle, or perhaps a riddle?

### Q: Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

Absolutely! It is *always* to your advantage to switch your choice in this scenario. This is the famous Monty Hall Problem, and it’s a surprisingly counterintuitive one.

Here's why switching doubles your chances of winning the car:

* **Initial Probability:** When you initially chose door number 1, you had a 1/3 chance of selecting the door with the car and a 2/3

chance of selecting a door with a goat.

* **The Host’s Action:** The host *knows* where the car is. By revealing a goat behind door 3, he’s not randomly choosing a door; he’s strategically eliminating one of the incorrect possibilities. This concentrates the remaining 2/3 probability onto the other unopened door (door 2).

* **Why it seems counterintuitive:** It feels like after the host reveals a goat, the odds become 50/50. However, the key is that the host's action *provides you with new information*.

**Therefore, switching doors doubles your probability of winning from 1/3 to 2/3.**

Would you like to try another probability or logic puzzle, or perhaps one involving a different type of game?

## Document the models response

The model generally performed well for the reasoning tasks even provided correct code example for a recursive function. The only logical conclusion that failed was that the switch is likely to be in the off state, therefore not pressed. But it answered the tricky question with the two guards and the gamehost question perfectly.

# Explore Ollama functionality

## ollama list

``` bash
gemma3:1b                  8648f39daa8f    815 MB    5 hours ago
llama3.1:8b                46e0c10c039e    4.9 GB    3 days ago
nomic-embed-text:latest    0a109f422b47    274 MB    3 days ago
gemma3:4b                  a2af6cc3eb7f    3.3 GB    3 days ago
gemma:latest               a72c7f4d0a15    5.0 GB    10 days ago
```

## ollama show gemma3:4b

Model

    architecture        gemma3

    parameters          4.3B

    context length      131072

    embedding length    2560

    quantization        Q4_K_M

  Capabilities

    completion

    vision

  Parameters

    temperature    1

    top_k          64

    top_p          0.95

    stop           "<end_of_turn>"

  License

    Gemma Terms of Use

    Last modified: February 21, 2024

### Ollama: Key Features and Limitations

Ollama is a platform for running large language models (LLMs) locally. It offers flexibility, privacy, and ease of use.

### 1. Features
Ollama allows users to run LLMs on their own devices. This improves privacy and removes the need for constant internet access after setup.
Users can adjust model settings or fine-tune them to better fit their specific tasks.
Ollama supports different types of LLMs. For example, it works with LLaMA and Stable Diffusion.
The tool offers both command-line and graphical interfaces, making it easy to use.
It can connect to APIs and other development tools for more advanced usage.

### 2. Limitations
Running LLMs locally requires a powerful computer. Older or weaker systems may perform poorly.
Models available for local use are often less powerful than cloud-based versions.
Users must manually download updates or new models.
Support for rare languages or niche topics is limited.
Ollama may not meet the needs of larger organizations that require advanced tools, support, or multi-user features.
Some systems may face setup or compatibility problems due to Ollama’s ongoing development.

## Part 2: Python Interface for LLM Reasoning

1. Implement the provided Python pseudo code interface for Ollama
2. Create a testing framework
3. Evaluate reasoning capabilities

___

To test the models functionallity, basic reasoning questions were prompted. The input prompts are denoten by "Q:" with the answers right below. The used model is gemma3:4b model

### Category: Simple deductive reasoning

### Q: All dogs are mammals. Fido is a dog. Is Fido a mammal?

<think>This is a classic example of a deductive argument, specifically a syllogism. The premises are:

1. All dogs are mammals.
2. Fido is a dog.

Therefore, the conclusion "Fido is a mammal" must be true. If all members of a group (dogs) have a certain characteristic (being mammals), and Fido is a member of that group, then Fido must also have that characteristic.</think>
<ans>Yes, Fido is a mammal.</ans>

### Category: Multi-Step Deductive Reasoning / Basic Inference

### Q: If the library is open, Sarah will go there. If Sarah goes to the library, she will borrow a history book. The library is open. Will Sarah borrow a history book?

<think>This is a classic example of a deductive reasoning problem, specifically a Modus Ponens argument. We have two premises:

1. If the library is open, Sarah will go there.
2. If Sarah goes to the library, she will borrow a history book.

We are also given that the library is open.

Let's apply the first premise: Since the library is open, Sarah will go there.

Then, applying the second premise: Since Sarah goes to the library, she will borrow a history book.

Therefore, Sarah will borrow a history book.</think>
<ans>Yes</ans>

### Category: Logic Puzzle with Constraints

### Q: There are three friends: Alex, Ben, and Chris. One of them is a doctor, one is a lawyer, and one is an engineer. Alex is not the lawyer. Ben is not the doctor. The engineer is younger than Chris. Ben is older than the lawyer. Who is the doctor, who is the lawyer, and who is the engineer?


Okay, let's break this logic puzzle down step-by-step.

<think>
We have three professions: doctor, lawyer, and engineer.
We know Alex is not the lawyer.
Ben is not the doctor.
The engineer is younger than Chris.
Ben is older than the lawyer.

Let's use a table to keep track of the possibilities:

| Person | Doctor | Lawyer | Engineer |
|---|---|---|---|
| Alex |  | No |  |
| Ben | No |  |  |
| Chris |  |  |  |

From "Ben is older than the lawyer," we know Ben cannot be the engineer (since the engineer has to be younger than Chris).

Also, since Ben is not the doctor, and Ben is older than the lawyer, Ben must be the lawyer.

| Person | Doctor | Lawyer | Engineer |
|---|---|---|---|
| Alex |  | No |  |
| Ben | No | Yes | No |
| Chris |  | No |  |

Now we know Ben is the lawyer. Since Alex is not the lawyer, Alex must be the engineer.  And since the engineer is younger than Chris, Chris must be the doctor.

Let's verify:

* Alex - Engineer (Younger than Chris)
* Ben - Lawyer
* Chris - Doctor

</think>
<ans>
Alex is the engineer, Ben is the lawyer, and Chris is the doctor.

## Category: Reasoning with Ambiguity and Abduction

### Q: You enter a room and find a shattered vase on the floor, an open window with a muddy paw print on the sill, and your cat looking unusually guilty nearby. Your dog, who loves to chase squirrels, is nowhere to be seen, but you remember letting him out into the garden (which is accessible from the window) about 10 minutes ago. What are the most plausible explanations for the broken vase? Rank them from most to least likely, and explain what additional information could help you confirm or deny each hypothesis

Okay, let's break down the possible explanations for the broken vase, ranking them by likelihood and considering what information we need to investigate further.

<think>Okay, this is a classic scenario with a good amount of potential for multiple interpretations. The key is to consider what each element – the broken vase, the muddy paw print, the guilty cat, and the dog’s recent release – could contribute to the scene.

**Ranking of Plausible Explanations (Most to Least Likely):**

1. **Dog's Mischief (High Probability):** The most likely explanation is that the dog, having been released into the garden, got into a frenzy chasing something – perhaps a squirrel, a bird, or even just a particularly interesting insect. In his excitement, he bumped into the vase, causing it to fall and break. The muddy paw print is consistent with this. The cat’s guilt is likely due to being startled by the commotion, or feeling responsible for allowing the dog to run loose.

2. **Cat's Accidental Action (Medium Probability):** It's possible the cat, in a playful or startled moment, knocked the vase over while investigating something. Cats are naturally curious and can be clumsy. The cat’s guilt could simply be a reflexive reaction to a perceived problem. This scenario relies more on the cat’s character, which can be variable.

3. **Small Animal Intrusion (Lower Probability):** It's possible a small animal (a squirrel, a rat, a bird) entered the room, causing the cat to react defensively, and in the process, the cat bumped into the vase. This is less likely because we have no immediate evidence of a small animal being present.

4. **Deliberate Act (Very Low Probability):** While possible, it's the least likely explanation. Someone deliberately broke the vase.  There's no obvious sign of forced entry or other indicators of a break-in.

</ans>
**Summary of Investigative Steps & Required Information:**

To confirm or deny these hypotheses, I would take the following steps and seek the following information:

1. **Examine the Vase Fragments:** Carefully collect all the fragments of the vase.  Looking for:
    * **Direction of impact:** This can help determine the force and angle of the fall, supporting the theory that it was knocked over.
    * **Material:** The type of material (ceramic, glass, etc.) could provide clues about the force involved.
    * **Presence of hair/fur:**  Further confirmation of the cat’s involvement.

2. **Observe the Dog:**  Immediately check the dog’s condition – is he injured?  Does he have mud on him?  Does he seem agitated or confused?  A closer examination of his paws/legs might reveal scratches or evidence of impact.

3. **Observe the Cat:** Thoroughly examine the cat for any injuries, signs of distress, or traces of mud.  Note his behavior – is he hiding, excessively cautious, or exhibiting unusual aggression?

4. **Inspect the Garden Area:**  Look for any signs of a struggle, disturbed plants, or evidence of a squirrel or other small animal.  Check if the dog has left any tracks beyond the window sill.

5. **Check for Security Cameras:** If available, review any security camera footage to see what happened leading up to the incident.

6. **Question the Cat:** (Okay, this is a bit silly, but observing the cat’s behavior in detail will give you a great insight into his possible role.)

By gathering this information, I could significantly narrow down the possibilities and determine the most likely cause of the broken vase.

## Category: Complex Counterfactual and Systems Reasoning

### Q: Imagine a world where, starting in the year 1900, the average global human lifespan instantly doubled, but global birth rates remained exactly the same as they were in our actual 1900. All other historical events and technological developments up to 1900 remain unchanged. Describe three major, distinct, and significant global consequences (social, economic, environmental, or political) that would likely be observable by the year 1950 in this alternate reality

Okay, let’s analyze this fascinating hypothetical scenario. A world where human lifespan doubles in 1900 with no change in birth rates is a profoundly destabilizing event. Here’s my breakdown of three major consequences by 1950, considering the immense shift in demographics and the unchanging birth rate:

<think>
This is a fantastic thought experiment! The key here is the *constant* birth rate. This means we’re not just talking about a larger population, we're talking about an *exponential* increase in the proportion of elderly people. It fundamentally alters everything. Let’s break down the potential consequences.

1. **Social - The Burden of Elders & Shifting Social Structures:** With a doubling of lifespan, the population would be dominated by older individuals. In our 1900, societies were already somewhat hierarchical, with elders holding considerable social and economic power. Now, this would become *extreme*. The elderly would constitute roughly 40-50% of the population. Traditional family structures would almost certainly collapse. The concept of ‘retirement’ as we understand it wouldn’t exist – people would remain productive (or at least appear so) for dramatically longer. There would be massive social tensions. Younger generations would face extreme competition for jobs and resources, compounded by a social system overwhelmingly shaped by the concerns and priorities of a much older demographic.  We might see a rise in age-based discrimination, potentially exacerbating existing social divisions. The value placed on youth and innovation would likely diminish significantly.

2. **Economic - Stagnation & Resource Strain:** The economic impact would be severe, primarily driven by labor force issues. With a huge proportion of the population being elderly and less able to work, there would be a severe shortage of labor. Industrial output would almost certainly stagnate. While there might be a boom in industries catering specifically to the elderly – healthcare, leisure, and luxury goods – this wouldn’t offset the overall economic decline.  Resource demands would also skyrocket. Food production would be under immense strain to feed a larger population, compounded by the increased need for energy and materials.  The pre-existing colonial economic systems would likely become even more exploitative, as industrialized nations would desperately need resources and cheap labor. There may be significant internal migration as people move to areas with fewer aged populations.

3. **Environmental - Accelerated Degradation:** This is arguably the most critical and concerning consequence. A doubling of lifespan, combined with a constant birth rate, would create an unparalleled strain on the planet's resources.  The existing levels of industrial pollution from 1900 would be dramatically amplified by a population that lives *much* longer and consumes far more during that extended lifespan. Deforestation would accelerate to meet increased demand for timber, agriculture and urban expansion. The sheer volume of waste produced would be overwhelming, creating enormous pollution problems. Furthermore, increased human activity would undoubtedly contribute to accelerated climate change, even if the scientific understanding of the issues was still nascent in 1950.  The ecological impact would be far worse than it was in our timeline, leading to widespread environmental degradation. </think>

<ans>
Here are three major, distinct, and significant global consequences observable by 1950 in this alternate reality:

1. **Social - Extreme Age-Based Stratification:** Society would be intensely stratified by age, with the elderly dominating political, economic, and social power structures. Traditional family structures would collapse, and significant age-based discrimination would likely be prevalent.

2. **Economic - Industrial Stagnation & Resource Depletion:** A severe labor shortage coupled with exponentially increasing resource demands would lead to industrial stagnation and accelerate the depletion of natural resources.

3. **Environmental - Accelerated Ecological Crisis:**  The combined effect of increased consumption and population size would trigger a dramatically accelerated ecological crisis, resulting in widespread pollution, deforestation, and contributing significantly to early stages of climate change. </ans>

### Evaluate reasoning capabilities:

### Q: How many r's are in the word strawberry?

<think></think>

The word "strawberry" contains the letter "r" five times.

<ans></ans>

## Analysis

The gemma3:1b model did not reason about the problem and provided a wrong answer.

### Test gemma3:4b

### Prompt: How many r's are in the word strawberry?

<think>The word is "strawberry". I need to count the number of times the letter 'r' appears in the word. Let's look at each letter: s, t, r, a, w, b, e, r, r, r, y.  There are three 'r's.</think>
<ans>3</ans>

## Analysis

The gemma3:4b model did reason about the problem by looking at each letter individually and afterwards counting the occurencies. Somehow it spelled out the letters of "berry" with 3 r's, but still got the answer right.

## Part 3: GRPO Fine-tuning

1. Set up the GRPO fine-tuning environment:
2. Evaluate the fine-tuned model: a. Ask the same question ("How many r's are in the  word strawberry?") to both the original and fine-tuned models

### Fine-tuning and setting up the ollama model from .gguf

The gemma3:1b model was finetuned using the provided Colab notebook (<https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb#scrollTo=5tkTF5Hmlhl->). After finetuning the model was saved as .gguf . In order to use it with ollama, a Modelfile had to be created. With `ollama show --modelfile gemma3:1b` the modelfile used for the native gemma3:1b could be used as a template. During finetuning, a system prompt was used:

```python
"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
```

This system prompt was also included in the modelfile. The modelfile then looked like the following:

```python
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

## Troubleshooting

While setting up this task, we encountered an issue while finetuning the model. Therefore we had to downgrade the versions of `unsloth` to `2025.5.7` and `unsloth-zoo` to `2025.5.8` using the following commands:

```bash
pip uninstall unsloth unsloth-zoo
pip install unsloth==2025.5.7 unsloth-zoo==2025.5.8
```

### Comparison

#### The question "How many r's are in the word strawberry? Explain why!" was given to both the native gemma3:1b and the fine-tuned gemma3:1b model ten times

The text "Explain why!" was added, because the reasoning was in every case "The word strawberry contains $x$ *r*'s"

### Native gemma3:1b

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

### Finetuned gemma3:1b

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

### Comparison

While the model, which is not finetuned, answers three times correctly, the finetuned model only provides one time the correct answer. Anyhow the result seems to be pretty random. Most of the time in the model, which is not finetuned, does not even give an explaination, though it was asked for one. This seems to be better with the finetuned model. The finetuned model does not give good reasons for why the word strawberry contains $x$ *r*'s, but at least it tries to explain.

## Part 4: Retrieval-Augmented Generation

1. Set up a RAG system using LangChain
2. Create a knowledge base:
3. Evaluate RAG performance

## Comparison, Rowing (sport)

Both queries were wrapped in a bigger prompts

```python

#without rag
"""
You are a helpful assistant. If unsure, say you don’t know.
question: {question}
"""

#with rag
"""
You are a helpful assistant. Use the context below to answer the question.
If unsure, say you don’t know.

<context>
{context}
</context>

Question: {question}
"""

```

### The question "who is faster yale or harvard" was given to both the native gemma3:4b and the gemma3:4b+rag

#### Native gemma3:4b

**Answer:**

<SOLUTION>Based on recent data and rankings, Yale is generally considered the faster school academically. Yale has a significantly higher graduation rate and a faster pace of completion for its degrees compared to Harvard.

However, both are incredibly prestigious universities!

Do you want me to elaborate on specific metrics that demonstrate this difference?

#### Rag: gemma3:4b

**Answer:**

<SOLUTION>Yale is faster. The context states Yale holds the upstream course record with a time of 18:35.8, while Harvard’s record is 18:22.4.

---

### The question "who won more consecutive regattas ?" was given to both the native gemma3:4b and the gemma3:4b+rag

#### Native gemma3:4b

**Answer:**

<SOLUTION>Please provide me with the context! I need the text you’re referring to in order to answer the question about who won more consecutive regattas.

#### Rag: gemma3:4b

**Answer:**

<SOLUTION>Harvard won more consecutive regattas, winning 18 consecutively from 1963 to 1980.

---

### The question "who won the last rowing regatta yale or harvard ? (2024)" was given to both the native gemma3:4b and the gemma3:4b+rag

#### Native gemma3:4b

**Answer:**

<SOLUTION> As of today, November 2, 2024, Harvard won the 2024 Harvard-Yale Regatta.

You can find the full results here: <https://harvardcrimson.com/news/2024/09/29/harvard-beats-yale-in-annual-rowing-race/>

#### Rag: gemma3:4b

**Answer:**

<SOLUTION> The text doesn’t provide information about the 2024 Yale-Harvard Regatta. It states there was no regatta in 2020 due to the COVID-19 pandemic, and doesn’t offer any details about recent races.

### Analysis

The system with the rag correctly reads most of the data and answers with the context of the input files. However it did not read the tables well, even if specifically prompted. Therefore it fails to answer the last question. The native model did answer it correctly but provided a dummy link.

When asked about facts not mentioned in the documents it fails to give an answer. `Who is the current pope` was answered with `I don't know`, while the native model answered with the outdated information `Francis`. This indicates that the prompting wrapper could be specified in such a way that the model only uses the document when the question is about the specified topics.

## REFERENCES

[1] Dominik, Edonis, and Nikola, Tower of hanoi, <https://github.com/sinedo/TowerOfHanoi>, Accessed:2025-05-20, 2025.

[2] Dominik, Edonis, and Nikola, Recordings, <https://tuwienacat-my.sharepoint.com/:f:/g/personal/e52004027_student_tuwien_ac_at/Eot8Zx_kbnxDv7esxmsF1BQBgh3UB-WVxEdsnCJfHhsdOw?e=1CMYyR>, Accessed: 2025-05-20, 2025.

[3] <https://de.wikipedia.org/wiki/Harvard_University>

[4] <https://de.wikipedia.org/wiki/Yale_University>

[5] <https://en.wikipedia.org/wiki/Rowing_(sport)>

[6] <https://en.wikipedia.org/wiki/Harvard%E2%80%93Yale_Regatta>
