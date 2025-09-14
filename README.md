# QA Retrieval Semantic Similarity
Implements semantic similarity methods to retrieve relevant responses in conversational QA. Identify, for each TEST prompt, the most similar prompt in TRAIN+DEV and return its corresponding response.
Implements multiple text representations (TF-IDF, static embeddings, and an optional hybrid) and evaluates DEV performance with BLEU before producing TEST CSV submissions.


## Project Overview

This project builds a **retrieval-based QA system**: given a new question (`user_prompt`) from TEST, we find the most similar question in TRAIN+DEV and output the **response ID** of that nearest neighbor.  
Tracks correspond to different text representations:

- **Track 1 — Discrete representations:** TF-IDF / n-grams + cosine similarity.  
- **Track 2 — Distributed static representations:** pretrained static embeddings (e.g., Word2Vec / FastText / Doc2Vec) + cosine similarity.  
- **Track 3 (Bonus) — Open:** any combination / alternative approach (e.g., hybrid TF-IDF + embeddings, SIF weighting, dimensionality reduction).

> Assessment is based on **BLEU** between retrieved responses and references (hidden for TEST). We tune on TRAIN+DEV and submit IDs for TEST.


## Data Format

Each record includes:
- `conversation_id` — unique ID  
- `user_prompt` — question text  
- `model_response` — response text (TRAIN/DEV only; TEST has no responses)

**Splits**
- **TRAIN & DEV:** prompts + responses (used to build and tune the retriever).
- **TEST:** prompts only (we output nearest TRAIN+DEV response IDs).


## Repository Structure

- `track 1.ipynb` — TF-IDF / CountVectorizer baseline with cosine similarity (Track 1).
- `track 2.ipynb` — Static embeddings (Word2Vec / FastText / Doc2Vec) with cosine similarity (Track 2).
- `track 3.ipynb` — Optional hybrid / open approach (Track 3).
- `dev_prompts.csv`, `train_prompts.csv` - prompts to train the model
- `dev_responses.csv`, `train_responses.csv` - responses to train the model
- `test_prompts.csv` - prompts only to test model


## Method Summaries 

**Track 1 — Discrete Text Representation**  
Used TF-IDF vectorization with unigrams to convert text to numerical representations. Preprocessed text by lowercasing, removing punctuation, and normalizing spaces. Calculated cosine similarity between test prompts and train+dev prompts to find the most semantically similar prompt and retrieve its corresponding response.

**Track 2 — Distributed Static Text Representation**  
Leveraged Google News Word2Vec pre-trained embeddings to convert preprocessed text into vector representations. Preprocessing involved lowercasing, removing punctuation, and lemmatizing tokens. Computed cosine similarity between test prompt vectors and train+dev prompt vectors to identify and retrieve the most semantically similar prompt's response.

**Track 3 — Open (Bonus)**  
Implemented a two-phase retrieval method combining TF-IDF and Sentence-BERT (SBERT). First used TF-IDF to identify top 100 candidate prompts from train+dev datasets. Then applied SBERT embeddings to re-rank candidates and select the most semantically similar prompt, ensuring robust and contextually aware response retrieval.



## How to Run
All steps for data loading, preprocessing, training, evaluation, and generating submissions are documented directly in the Jupyter notebooks:

- `track 1.ipynb` — TF-IDF baseline (Track 1)  
- `track 2.ipynb` — Static embeddings (Track 2)  
- `track 3.ipynb` — Hybrid / open approach (Track 3, optional)  

Follow the instructions in each notebook to reproduce the results and create the submission files.


## Credits
Assignment developed as part of coursework in the Natural Language Processing course at Bocconi University, academic year 2024-2025.

## Contact
For questions: luca.milani2@studbocconi.it
