GLiFormer and Encoder-Decoder Architecture
This repository documents the architecture and operational principles of the GLiFormer Large v1 model (knowledgator/gliformer-large-v1), comparing its encoder-based foundation with decoder-only Large Language Models (LLMs).   
MD
Quickstart & Local Setup
1. Clone & Environment
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
2. Dependencies
Bash
pip install torch torchvision torchaudio
pip install gliformer -U
3. Minimal Inference Script (test_gliformer.py)
Python
import torch
from gliformer import GLiFormer

model = GLiFormer.from_pretrained(
    "knowledgator/gliformer-large-v1", load_tokenizer=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

text = "Alice works at Acme in London."
labels = ["person", "organization", "location"]

entities = model.predict_entities(text, labels, threshold=0.5)
print(entities)
1. Encoders vs. Decoders
Encoder (Bidirectional): Evaluates all tokens simultaneously (past and future context) to build comprehensive semantic representations. Ideal for understanding, tagging, and extraction tasks (e.g., DeBERTa, BERT).   
MD
+ 1
Decoder (Unidirectional / Causal): Generates tokens sequentially from left to right. Future positions are mathematically masked with -inf to set attention weights to zero, preventing the model from trivial copying during generation.   
MD
+ 1
2. Attention & Feed-Forward Mechanism
Attention: Computes pairwise dot-product compatibility scores between token vectors, applies Softmax normalization, and computes weighted averages to enrich representations with context.   
MD
Bidirectional vs. Causal Attention Matrix:
Encoder: All tokens can attend to all tokens.   
MD
Decoder: Tokens can only attend to prior and current tokens; future cells are masked.   
MD
Feed-Forward Blocks (MLP): Expands representations (e.g., 1024→4096→1024) to apply non-linear transformations independently to each token.   
MD
Pattern Recognition: In sentences like "Alice works at Acme", classification as a person emerges through learned statistical patterns (capitalization, sentence position, preceding verbs) rather than hard-coded heuristics.   
MD
3. DeBERTa Backbone & Disentangled Attention
DeBERTa stacks multiple transformer encoder blocks (e.g., 24 layers in Large). It separates token semantics using disentangled attention:   
MD
Content Vector: Encodes the intrinsic meaning of the token.   
MD
Position Vector: Encodes relative positioning within the sequence.   
MD
Attention weights are calculated by explicitly handling content-to-content, content-to-position, and position-to-content components separately before merging.   
MD
4. GLiFormer's Five Specialized Heads
All task heads share the identical DeBERTa encoder representations, projecting them into specific decision boundaries via lightweight linear layers and activation functions:   
MD
NER Head: Performs span-level entity detection independently (predict_entities) using a confidence threshold.   
MD
Classification Head: Aggregates context across the full sequence to assign categorical labels or group predictions (classify).   
MD
Joint Relation Head: First extracts candidate entities via NER, then pairs them to classify formal relational triples (e.g., Alice - works_at - Acme) (inference).   
MD
Structuring Head: Extracts entities directly into predefined key-value formats (Python dict) or nested schemas using Pydantic (structure).   
MD
Embedding Head: Maps arbitrary sequences into fixed 1024-dimensional continuous vectors (embed_text) for similarity search via cosine similarity.   
MD
5. Multi-Task Training & Zero-Shot Capabilities
Multi-Task Batch Interleaving: Samples across tasks are mixed during training to prevent catastrophic forgetting. Gradients update only the shared encoder and the active task head during that specific forward pass.   
MD
+ 1
Constrained Zero-Shot: While Large Language Models interpret open-ended natural language prompts, GLiFormer operates through fixed programmatic function calls with dynamic, open-schema label names. It evaluates compatibility against labels unseen during training without requiring fine-tuning.   
MD
+ 1
Domain Transfer Notes: Out-of-domain benchmark results (such as CrossNER) indicate strong transfer capabilities, though pre-training exposure to related concepts means metrics should be interpreted with empirical rigor rather than pure zero-shot isolation.   
MD
6. Evaluation Benchmarks & Metrics
F1 / Precision / Recall: Summarizes exact boundary and label accuracy.   
MD
Macro-F1 vs. Weighted-F1: Macro-F1 averages across classes equally, while Weighted-F1 factors in class frequencies. A large gap between the two highlights lower performance on sparse minority classes.   
MD
+ 1
Gold Relations: Ground-truth human annotations used as the reference target in benchmarks like DocRED.   
MD
Multimodal Limitations: This checkpoint possesses no visual (vision) head and operates strictly on text sequences.   
MD
7. Architectural Comparison: NLU vs. LLMs
Property	NLU (GLiFormer)	Generative LLMs
Architecture	
Encoder-only  
MD

Decoder-only  
MD

Attention	
Bidirectional  
MD

Unidirectional (Causal)  
MD

Primary Role	
Sequence understanding & tagging 
MD

Text generation & continuation 
MD

Output Type	
Deterministic (tensors, dicts)  
MD

Free-form natural language 
MD

Compute Passes	
1 pass per sequence  
MD

1 pass per generated token 
MD

Interface	
Programmatic API functions  
MD

Natural language prompts  
MD

Parameter Scale	
Hundreds of millions  
MD

Billions to hundreds of billions 
MD

Summary
GLiFormer integrates five downstream NLU heads over a DeBERTa encoder backbone, processing text bidirectionally to achieve efficient zero-shot extraction, classification, and schema structuring. It delivers fast, structured, and deterministic outputs ideal for information extraction pipelines where generative LLM overhead is unnecessary.   
MD
+ 1
Hasan Ali Kınaş

