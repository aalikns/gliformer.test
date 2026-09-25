# GLiFormer and Encoder-Decoder Architecture

This document explains how the GLiFormer Large v1 model shared at [huggingface.co/knowledgator/gliformer-large-v1](https://huggingface.co/knowledgator/gliformer-large-v1) works, its encoder-decoder architecture, and how it differs from Large Language Models (LLMs).

## Table of Contents

1. [Local Installation and Setup](#local-installation-and-setup)
2. [What Are Encoders and Decoders](#1-what-are-encoders-and-decoders)
3. [The Attention Mechanism](#2-the-attention-mechanism)
4. [DeBERTa Architecture and the Concept of Parameters](#3-deberta-architecture-and-the-concept-of-parameters)
5. [GLiFormer's Five Task Heads](#4-gliformers-five-task-heads)
6. [Multi-Task Training](#5-multi-task-training)
7. [The Concept of Zero-Shot](#6-the-concept-of-zero-shot)
8. [Terms in Evaluation Tables](#7-terms-in-evaluation-tables)
9. [Visual Analysis Capability of the Model](#8-visual-analysis-capability-of-the-model)
10. [Differences Between NLU and LLMs](#9-differences-between-nlu-and-llms)
11. [Summary](#summary)

---

## Local Installation and Setup

Follow these steps to run GLiFormer on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
2. Create and Activate a Virtual Environment
Bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
Install PyTorch and the GLiFormer package:
Bash
pip install torch torchvision torchaudio
pip install gliformer -U
4. Minimal Working Example (test_gliformer.py)
Python
import torch
from gliformer import GLiFormer

# Load the pretrained model
model = GLiFormer.from_pretrained(
    "knowledgator/gliformer-large-v1", load_tokenizer=True
)

# Select GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# Run Zero-Shot NER
text = "Alice works at Acme in London."
labels = ["person", "organization", "location"]

entities = model.predict_entities(text, labels, threshold=0.5)
print(entities)
Run the script from your terminal:
Bash
python3 test_gliformer.py
1. What Are Encoders and Decoders
A model processing text can perform two distinct tasks: understanding the text or generating its continuation. These two tasks require different architectures.
An Encoder, given a sentence, assigns meaning to each word by looking at the entire sentence (both preceding and succeeding words). This is called bidirectional processing.
A Decoder generates the continuation of a text word by word, in sequential order. When generating each new word, it can only see the words that came before it; it cannot see the words that have not been generated yet. This is called unidirectional (causal) processing.
The decoder's inability to see the future is an intentional design. If the decoder were allowed to see the future during training, the model would simply see and copy the real answer without learning anything. Therefore, in the decoder's attention computation, the cells corresponding to words that have not yet been produced are mathematically set to -infinity, which automatically resets the weights of those cells to zero.
When a message is written to a decoder model (for example, a chat assistant), that message is considered the model's history because it was written before the model responded. The model can look at this entire history while generating a response. A separate encoder is not required because the message and the response are parts of the same, single text stream. This architecture was originally designed for tasks such as translation (between a source language and a target language); conversational models only use the decoder portion since they do not have this requirement. Models designed for understanding tasks (such as DeBERTa and BERT) use only the encoder portion.
2. The Attention Mechanism
Attention is the mechanism that calculates how much importance a word should assign to the other words in the sentence.
Each word in a sentence is first converted into a vector (a list of numbers). In reality, these vectors are 768-1024 dimensional; they can be shown as 3-dimensional for illustrative purposes:
Ali    = [0.2, 0.8, 0.1]
gitti  = [0.5, 0.3, 0.9]
okula  = [0.1, 0.6, 0.4]
When processing the word "gitti", dot products are calculated with other words:
gitti · Ali   = (0.5×0.2)+(0.3×0.8)+(0.9×0.1) = 0.43
gitti · okula = (0.5×0.1)+(0.3×0.6)+(0.9×0.4) = 0.59
gitti · gitti = (0.5×0.5)+(0.3×0.3)+(0.9×0.9) = 1.15
These numbers are converted into percentages using the softmax operation:
Attention to Ali:    20%
Attention to gitti:  52%
Attention to okula:  28%
A weighted average of all word vectors is computed using these percentages, producing the new, context-rich vector for the word "gitti". A word assigning a high weight to itself is an expected outcome, because a word's own meaning is its primary source of information during processing.
In an encoder, every word can look at every other word in the attention matrix:
Ali	gitti	okula
Ali	yes	yes	yes
gitti	yes	yes	yes
okula	yes	yes	yes
In a decoder, only the past context can be accessed:
Ali	gitti	okula
Ali	yes	no	no
gitti	yes	yes	no
okula	yes	yes	yes
Cells marked as "no" are excluded from the computation.
Example: How "Alice" is determined to be a person in "Alice works at Acme"
The word "Alice" is converted into a vector.
Through attention, a new contextualized vector is constructed by attending to the words "works", "at", and "Acme".
This process repeats throughout the model (approximately 24 layers in DeBERTa Large), with the vector gaining more contextual meaning at each layer.
The vector exiting the final layer is sent to the NER head.
The NER head multiplies this vector by a small matrix to produce four numbers: probabilities for person, organization, location, and none.
The values inside this matrix were learned from millions of training examples. Patterns such as words starting with a capital letter at the beginning of a sentence before a verb usually being a person are captured in these weights.
Consequently, the probability for person emerges high.
Nowhere in this process is there an explicit written rule like "if it starts with a capital letter, it is a person." This is entirely a statistical pattern.
In each layer, after the attention block, there is also a feed-forward block (MLP - Multi-Layer Perceptron):
Input (1024 numbers) -> matrix (expand from 1024 to 4096) -> activation 
                     -> matrix (project from 4096 back to 1024) -> Output (1024 numbers)
While attention models relationships between words, the feed-forward block processes each word individually and more deeply.
3. DeBERTa Architecture and the Concept of Parameters
DeBERTa consists of layers stacked sequentially on top of each other:
Input text
    -> Embedding layer (converts words into initial vectors)
    -> Layer 1 (Self-attention + Feed-forward)
    -> Layer 2
    -> ...
    -> Layer 24
    -> Final output vectors
The feature distinguishing DeBERTa from BERT is disentangled attention: the content (meaning) and position (placement in the sentence) of each word are kept in separate vectors; when attention is calculated, these two components are processed separately and combined afterward.
4. GLiFormer's Five Task Heads
GLiFormer was trained by mounting five different task heads onto a single DeBERTa encoder. All heads are fed from the identical encoder output, and each translates this output into a different objective.
NER Head
NER stands for Named Entity Recognition. It makes independent decisions for each word individually.
Python
entities = model.predict_entities(
    "Alice works at Acme in London.",
    ["person", "organization", "location"],
    threshold=0.5,
)
Output: Alice is labeled as person, Acme as organization, and London as location. Each result includes text, label, start position, end position, and a confidence score.
The threshold is the minimum confidence score (between 0 and 1) required to consider a prediction valid.
Classification Head
Instead of evaluating words individually, it examines the entire sentence to produce a single classification decision.
Python
predictions = model.classify(
    "The new search feature is fast and easy to use.",
    ["positive", "negative", "neutral"],
    threshold=0.5,
)
Named groups are also supported; for example, sentiment and topic classification can be performed concurrently in a single call.
Joint Relation Head
Unlike NER's "what is this" question, this head evaluates two entities together and queries whether a specific relation exists between them.
Python
results = model.inference(
    "Alice works at Acme.",
    joint_relations={
        "employment": {
            "entities": ["person", "organization"],
            "relations": ["works_at"],
        }
    },
    threshold=0.5,
)
The output is a triple formatted as Alice - works_at - Acme.
In this process, NER runs first to determine the entity types for Alice and Acme individually. Subsequently, these discovered entities are paired up to identify relations between them. While NER already utilizes sentence context through attention, it does not output a formal relation label; it simply marks a word as an organization. Identifying and formalizing the relation into a structured triple is a subsequent step built on top of the NER extractions.
Structuring Head
Places information into a predefined schema.
Python
records = model.structure(
    "Alice works at Acme.",
    {"employee": ["name", "company"]},
)
# Output: {'employee': [{'name': 'Alice', 'company': 'Acme'}]}
JSON is a format that organizes data into key-value pairs, making it both human-readable and easily processed by software. Its equivalent in Python is the dict data type. The reason for converting free text into JSON is to allow downstream software to consume this data directly without needing custom parsers.
Nested schemas are also supported. Multi-layered hierarchies, such as departments within a company and employees within departments, can be defined and populated using the Pydantic library.
Embedding Head
Converts the semantic meaning of a sentence into a fixed-size numerical vector.
Python
embeddings = model.embed_text([
    "A scientist works in a laboratory.",
    "A researcher conducts an experiment.",
])
print(embeddings.shape)  # torch.Size([2, 1024])
Each sentence is transformed into a vector consisting of 1024 numbers. The semantic similarity between two sentences can be measured using cosine similarity between their vectors; values range from 0 to 1, with values closer to 1 indicating matching semantics.
Technical Meaning of Adding a Head
A head is a structure that takes the encoder output and transforms it into a specific decision via a small sequence of additional matrices:
Encoder output (1024 numbers)
    -> matrix (from 1024 to 256)
    -> activation
    -> matrix (from 256 to 4, where 4 = number of categories)
    -> softmax
    -> person: 2%, organization: 94%, location: 3%, none: 1%
This is a lightweight structure relative to the encoder; a separate head is attached for each distinct task.
5. Multi-Task Training
During training, examples belonging to different tasks (NER, classification, relation, structuring) are presented in mixed order. If training were carried out exclusively on one task and then sequentially on another, updating encoder weights for the second task could degrade what was learned for the first. This condition is called catastrophic forgetting. Shuffling examples prevents the encoder from overfitting to a single task and forgetting others.
When an NER sample arrives, only the encoder and the NER head are utilized; the other heads remain inactive and receive no weight updates during that step. When a classification sample arrives, only the encoder and classification head receive updates. This behavior stems from the mathematical structure of backpropagation: error propagates backward only along the computational graph utilized during that iteration. The encoder updates on every step because every task relies on it, whereas individual heads update only during their respective tasks.
6. The Concept of Zero-Shot
Zero-shot means obtaining predictions from a model without providing task-specific training examples, relying solely on task descriptions or label definitions.
When a large language model is instructed to "classify this sentence", it reads and interprets this instruction in natural language and generates an open-ended output. Because it has seen billions of diverse instruction-response examples during training, it can comprehend entirely novel instructions and produce relevant outputs.
In GLiFormer, instructions are not written in natural language; objectives are specified via direct function calls:
Python
model.classify(text, ["positive", "negative"])  # works
model.classify(text, "find sentiment")  # does not work
GLiFormer's zero-shot capability means that while the functional interface (classify, predict_entities, etc.) remains fixed, the label names provided to that interface can vary dynamically. The model can interpret and predict label names it has never seen during training, without requiring fine-tuning.
The distinction can be summarized as follows: Large language models comprehend both what the task is and how to execute it from arbitrary natural language. In GLiFormer, the task scope is fixed in code, and the model resolves only the semantics of target label names. This represents a narrower form of zero-shot compared to the flexibility found in LLMs, but it provides a major advancement over legacy NER models locked to hard-coded categories.
In the repository documentation, benchmark scores on datasets like CrossNER are presented to demonstrate out-of-domain transfer success. However, notes specify that reported transfer groups do not guarantee the domain was completely absent during pre-training. While the model has not observed the specific evaluation set, it may have gained familiarity with related concepts from other pre-training datasets. Consequently, these metrics should be interpreted with realistic caution rather than as entirely pure zero-shot evaluations.
7. Terms in Evaluation Tables
F1 Score summarizes precision and recall into a single scalar between 0 and 100.
Precision indicates the proportion of items marked as positive by the model that are actually correct.
Recall indicates the proportion of actual reference entities correctly identified by the model.
Strict entity F1 is a rigorous evaluation mode requiring exact matches on both token boundaries and entity labels for a prediction to count as correct.
Macro-F1 is the unweighted arithmetic mean of F1 scores calculated independently for each class. For example, if the positive class has 900 samples with 0.95 F1, negative has 80 samples with 0.60 F1, and neutral has 20 samples with 0.30 F1, Macro-F1 is simply: (0.95 + 0.60 + 0.30) / 3 = 0.617.
Weighted-F1 weights each class's F1 score based on its sample count: (900×0.95 + 80×0.60 + 20×0.30) / 1000 = 0.909. If Weighted-F1 is high while Macro-F1 remains low, it indicates that the model excels on dominant majority classes but underperforms on minority classes.
The term "gold relations" in relation extraction benchmarks refers to ground-truth reference data annotated by humans. For example, reporting 6003 gold relations in DocRED indicates there are 6003 true relation instances in the evaluation split that the model must identify.
In the DocRED results, metrics are reported as precision 29.90, recall 8.13, micro-F1 12.78, and macro-F1 3.64. This indicates that while roughly 30% of relationships predicted by the model are accurate, it captures only 8% of all existing target relationships. Macro-F1 falling significantly lower than micro-F1 shows the model struggles considerably on rare relation types.
8. Visual Analysis Capability of the Model
This model cannot perform visual analysis. The limitations section of the model documentation explicitly states that this checkpoint contains no visual (vision) head.
9. Differences Between NLU and LLMs
NLU (GLiFormer)	LLMs (Large Language Models)
Core Architecture	Encoder-only	Decoder-only
Attention Direction	Bidirectional	Unidirectional (Causal)
Primary Task	Analyzes and labels existing text	Generates new text
Output Type	Fixed-format	Free-form, fluent text
Required Forward Passes	One (entire text processed in a single pass)	One pass per generated token
Instruction Following	None, guided via function calls	Yes, natural language instructions parsed
Conversational Ability	No	Yes
Parameter Scale	In the hundreds of millions	Billions to hundreds of billions
NLU models are preferred when millions of sentences need to be labeled quickly and cheaply, when outputs must conform to a deterministic format, when low-cost local execution is required, and for structured data extraction from documents.
Large language models are preferred for conversational assistance, question answering, open-ended explanations, tasks where instructions must be parsed dynamically, creative writing, and complex reasoning.
A large language model can theoretically handle entity extraction, but it incurs a slower generation cycle for every request, and schema compliance is not guaranteed. A small encoder model executes the same task much faster, at minimal compute cost, and with guaranteed structural adherence. Therefore, these two model families are not replacements for one another; they are designed for different operational requirements.
Summary
GLiFormer is a multi-task model constructed by attaching five distinct task heads (NER, classification, relation extraction, structuring, embedding) to a single DeBERTa encoder. Its encoder architecture analyzes text using bidirectional attention, distinguishing it from unidirectional, generative decoder-based LLMs. The model's zero-shot capability enables it to recognize unseen entity and classification labels, representing a more constrained mechanism compared to the open-ended instruction-following capabilities of LLMs. Benchmark results indicate that the model performs strongly on classification and structuring tasks while showing lower metrics in relation extraction, with all evaluations conducted on English benchmarks.
Hasan Ali Kınaş

