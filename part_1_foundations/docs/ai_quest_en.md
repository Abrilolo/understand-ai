AI Quest
--------

# Quest Description

- Research and Software Development Project

## Objective

- Build a complete mental model of modern AI by constructing it yourself, one layer at a time, starting from a single neuron and ending with a tool-using agent.
- Understand the difference between AI, Machine Learning, Deep Learning, and Large Language Models, not as buzzwords but as concrete systems you have implemented.
- Demystify the transformer and the "attention" mechanism by building a tiny one from scratch.
- Answer, with evidence you have gathered yourself, the question: "Are LLMs really just very big autocompletes?"

## Required Skills

### Technical:

- Python (the whole quest is in Python)
- Comfort with the command line and Git

### Knowledge:

- Basic linear algebra (vectors, matrices, dot products) is recommended but can be learned along the way
- Basic calculus (what a derivative is) helps for the backpropagation part

### Experience:

- No previous machine learning experience is required. This quest seeks to provide the full road for acquiring such experience.

### Other:

- Curiosity and patience
- Skills to independently research public information on the internet

## Technical Level

| Level | Description                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | No experience                                                                                                                                           |
| 1     | Basic programming knowledge and functional code experience                                                                                              |
| 2     | Basic programming skills and practical experience in a specific technology or framework                                                                 |
| 3     | Intermediate programming skills and experience in a specific area                                                                                       |
| 4     | Advanced programming skills and expertise in a specific domain                                                                                          |
| 5     | This is a project for advanced users who have deep specialized knowledge and extensive practical experience and can complete the project by themselves. |

> Difficulty level: [0-5] (1 is for beginners, 5 is for advanced users)

- Difficulty: 3-4 (the early parts are approachable for beginners, the later parts reach into advanced territory)

## Project Content

### Detailed Task Description:

The quest is split into eight parts. Each part builds on the previous one, so complete them in order. Every part has its own concrete deliverable. The detailed instructions for each part are in the `# issue` section below.

### Workflow:

Promote the project using the Scrum method.

- GitLab
- Redmine

AI assistants are allowed, including for writing code, as long as you can prove you understand all of the concepts behind every implementation. If you cannot explain why a piece of code works, you have not finished that part.

### Deliverables:

Push the programs, data, and documentation to a git repository.

- The code for each of the eight parts, in clearly separated folders
- A presentation on the history of Artificial Intelligence (Part I)
- A short write-up (a README per part is fine) explaining what you built and what you learned
- A final one-page reflection answering "Are LLMs really just big autocompletes?"

## Skills and Experience You'll Gain

### Skills:

- Implementing a neural network and backpropagation from scratch
- Using TensorFlow or PyTorch and understanding tensors
- Building a tokenizer and a small language model
- Running local open-source models with Ollama and giving them tools

### Experience:

- Turning a research topic into a working demo
- Building intuition for how modern AI systems actually work under the hood
- Technical documentation and presentation

### Other Benefits:

- The confidence that comes from having built things instead of importing them

### What You'll Learn Even If You Fail

- Where your math or programming gaps are, and how to close them
- Why "it works in a library" and "I understand why it works" are very different milestones
- How to break a huge, intimidating topic into small buildable steps

# issue

Work through the parts in order. Each part lists the questions you should be able to answer by the end and the deliverable you must produce.

## Part I: The Ground

![History of Neural Networks](https://media.licdn.com/dms/image/v2/C5612AQFxWfR5RfikJQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1520042220851?e=2147483647&v=beta&t=NK6oBrYT2_d6LhonZ1tBCh17NJuwVbAzLXU1INE9Pk4)

Let's start at the very bottom. Before touching a library, we build a neuron out of plain Python and understand every number that flows through it.

By the end of this part we should be able to answer:

- [ ] What is a neuron? What is a weight? What is a bias? What is an activation function?
- [ ] What is a neural network, and how is it more than the sum of its neurons?
- [ ] What kinds of problems are neural networks good at, and what kinds are they bad at?
- [ ] How do networks learn? What is backpropagation, in our own words?
- [ ] What are the common weight-adjustment (learning) methods, for example gradient descent and its variants?
- [ ] What is the difference between AI, Machine Learning, Deep Learning, and an LLM?
- [ ] Why can a single neuron learn AND, but not XOR?

Tasks:

- Implement a single neuron and then a small multi-layer network in pure Python (no ML libraries), including a working backpropagation loop.
- Build a single neuron that computes the logic gate **AND**. Find the weights and the bias that make it produce the right answer for all four inputs, either by hand or by training it. Plot the four input points on a square and draw the line the neuron is using to separate them.
- Now try the same for **XOR** with a single neuron. Plot its four points too. We will not succeed, and that is the point. Then build the smallest network that does succeed.
- Explain, in our own words, why one neuron is enough for AND but not for XOR. The two plots should make the answer visible before we write a single sentence about it. Hint: think about what shape a weighted sum plus a threshold can draw, and how many of those shapes we get once we add a hidden layer.
- Train the network on a small public tabular dataset. The recommended one is the [Iris flower dataset](https://archive.ics.uci.edu/dataset/53/iris): 150 rows, 4 numeric features, 3 flower species, no missing values, so it trains in seconds even on a from-scratch network and it is easy to tell whether the result is good. If we want something messier afterwards, Titanic or California Housing are good next steps.
- Research the history of Artificial Intelligence, from the 1940s to today, and prepare a presentation on it.

Deliverable: the from-scratch network code, the AND and XOR gates with their plots and our written explanation, the trained simple predictor, and the AI-history presentation.

### Video references

- [Neural networks, by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): the clearest visual explanation there is of what a network is actually computing and what backpropagation actually does. Watching this series before writing code will save us hours of confusion.

## Part II: Learning Without a Teacher, Evolving Flappy Birds

![Flappy Bird](https://upload.wikimedia.org/wikipedia/en/0/0a/Flappy_Bird_icon.png)

In Part I the network learned because we told it the right answer and pushed the error backwards. But what if nobody knows the right answer? Here there is no labelled dataset and no gradient, only survival. We are going to let a population of birds play Flappy Bird, kill off the bad ones, and breed the good ones, until one of them plays forever.

By the end of this part we should be able to answer:

- [ ] What is a genetic algorithm? What do fitness, selection, crossover, and mutation mean?
- [ ] Why does evolution work here, where plain backpropagation is awkward to apply?
- [ ] What happens when the mutation rate is too high? What happens when it is too low?
- [ ] Is this still "learning"? How would we defend that answer?

Tasks:

- Build a minimal Flappy Bird in pure Python with `pygame`. That library is the only new dependency, and it is only for drawing. No ML library.
- Give every bird a copy of the network from Part I as its brain. Suggested inputs: the bird's height, its vertical velocity, the horizontal distance to the next pipe, and the vertical distance to the centre of the next gap. One output: flap or do not flap.
- Run a whole generation at once, 100 birds or more, all flying on the same pipes. A bird's fitness is how far it gets before it dies.
- When the generation is dead, keep the best birds, copy their weights into the next generation with small random changes, and run it again. Repeat.
- Record the best and the average fitness of each generation and plot the curve. Describe its shape: where does it jump, where does it plateau, and why?
- Experiment: change the population size and the mutation rate, and see what each one does to the curve.
- Bonus: draw the best bird's network on screen next to the game, so we can watch its brain fire while it flies. This is the visualizer from Part III, so it may be easier to come back to this after that part.

Deliverable: the game and the evolution loop, the fitness-over-generations plot, and our notes on what population size and mutation rate did to the results.

### Video references

- [Evolution Simulator (Part 1/4)](https://www.youtube.com/watch?v=GOFws_hhZs8): a population of random blobs of nodes and muscles that is bred, generation after generation, until it learns to walk. Same idea we are building here, no backpropagation anywhere, and it makes the selection-and-mutation loop very easy to picture.

## Part III: A Neural Network Visualizer

![Neural Network Visualizer](https://www.joshgracie.com/nnviz.png)

A network we can see is a network we understand. Here we build a visualizer for the network we made in Part I.

Tasks:

- Draw the network: nodes for neurons, edges for weights.
- Show the weights and activations as data flows through, so we can watch a prediction happen.
- Bonus: animate the weights changing during training.

Deliverable: an interactive or animated visualizer of our Part I network.

## Part IV: Plain Python vs Specialized Tensor Libraries

![Word embeddings visualized with PCA and t-SNE](https://towardsdatascience.com/wp-content/uploads/2020/10/1bWwsXa36ugemh9YMcLdV9w.png)

Now we should find out why real AI is not written in plain Python lists, by benchmarking our hand-rolled math against a specialized library.

By the end of this part we should be able to answer:

- [ ] What is a tensor? How is it different from a vector?
- [ ] What is an embedding? What is a token?

Tasks:

- Re-implement a core operation from Part I (for example the forward pass) two ways: once with plain Python data structures, once with TensorFlow or PyTorch tensors.
- Benchmark them, measure and compare the speed, and explain the difference we observe.

Deliverable: the benchmark code and a short write-up of the results with our explanation.

### Video references

- [How AI could help us talk to animals](https://www.youtube.com/watch?v=7PgSanU_VpQ): what embeddings are for, taken to its most ambitious conclusion. Researchers are turning whale and elephant calls into vectors and looking for structure in the resulting space, the same way word embeddings put related words near each other.

## Part V: Build a Tokenizer

![The art of tokenization: breaking text down for AI](https://towardsdatascience.com/wp-content/uploads/2024/09/1QVXvydRMEWTWiUP42bYBAg.png)

Language models do not read text, they read tokens. Here we build the thing that turns text into tokens.

Tasks:

- Implement a tokenizer that turns text into tokens and back again.
- Start simple (split on characters or words), then improve it toward a subword approach such as Byte Pair Encoding.
- Run the tokenizer on a few example sentences, including at least one in Japanese, and see what breaks.

Deliverable: our tokenizer, with examples of it encoding and decoding text.

## Part VI: Our First Language Model

![What is autocomplete?](https://limeup.io/wp-content/uploads/2024/01/Auto-Complete.png)

Here we reuse the neural network from Part I and the tokenizer from Part V to make a network that predicts the next token.

Tasks:

- Get a small plain-text corpus to train on. We do not need anything huge. Good free options: the "Tiny Shakespeare" dataset (a single ~1 MB text file, widely mirrored online), any book from [Project Gutenberg](https://www.gutenberg.org/) as a plain `.txt`, or even our own text file (chat logs, a favorite article). A few hundred kilobytes of text is plenty for a first model.
- Build a minimal *neural* language model that predicts the next token, using the tokenizer from Part V to turn that text into tokens and the network from Part I to learn the prediction (for example a tiny character-level or token-level next-token predictor).
- Have the neural model generate text and observe the results.

Once the implementation works, let's answer the following question:

- [ ] Is the implemented neural language model different from just counting how often words follow each other and creating a map?

Deliverable: the language model, a few text samples it generated, and our notes on how it behaves.

## Part VII: is Attention really All We Need?

![Transformer architecture](https://media.licdn.com/dms/image/v2/D4D12AQEj4ADRPfqFyw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1701459621193?e=2147483647&v=beta&t=38s3cFvuDOl-PaRp_pZqfrWZaXq19Nt9cfnHLD-wKmQ)

![Optimus Prime](https://i.pinimg.com/736x/66/06/b8/6606b80ebc623553005f17d05c50a604.jpg)

We will now build a tiny transformer to understand better the concept of "attention".

Tasks:

- Implement a minimal transformer, including a self-attention mechanism, and explain in our own words what attention computes and why it matters.
- Train it on a small text dataset and generate text from it.
- Compare its output to our Part VI model and describe what changed.

Deliverable: our transformer implementation, generated samples, and a short explanation of attention.

## Part VIII: Going Agentic

![Running open-source LLMs locally with Ollama](https://res.cloudinary.com/dry8rzbyx/image/fetch/s--k9EzKtnt--/f_auto/q_auto/c_scale,w_550/https://www.knime.com/sites/default/files/public/2024-07/ollama-llama3-knime-550-100.jpg)

We have built the pieces. Now we use a real model and make it act.

Tasks:

- Download Ollama and run a lightweight open-source model locally.
- Give the model one or more tools (for example a calculator, a web search, or a file reader) and let it decide when to call them.
- Build a small demo where the model uses its tools to complete a task it could not do on its own.

Deliverable: our agent demo, plus notes on what worked, what failed, and where the model needed help.

## Final Reflection: are LLMs Really Just Very Big Autocompletes?

We have now built the whole stack ourselves: a neuron, a network, a population of birds that learned to fly without a teacher, a tokenizer, a neural language model, a transformer, and an agent. So we should have earned the right to answer the following question.

- [ ] Are LLMs really just very big autocompletes?

Write a reflection. Argue both sides honestly: in what sense is "just autocomplete" true, and in what sense is it misleading?

## Random References That Could Be Interesting

Nothing here is required for any part of the quest. These are just things that might be worth watching.

- [Growing Living Rat Neurons To Play... DOOM?](https://www.youtube.com/watch?v=bEXefdbQDjw): real biological neurons in a dish, wired up as a game controller. A good reminder of where the word "neuron" came from, and of how much our version simplifies.
- [Simulating the Evolution of Aggression](https://www.youtube.com/watch?v=YNMkADpvO4w): the same evolution loop from Part II, but the creatures are choosing whether to share or to take. Cooperation and aggression appear on their own, without anyone programming either one in.
