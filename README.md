# Job Matcher

## Overview
A candidate-to-job matching system that pairs candidate profiles with YC startup job listings. Candidates select or paste their profile in json format, receive ranked recommendations, and iteratively refine results through natural language feedback parsed by Claude.

## Setup
1. run `cd backend` in the terminal
2. run `pip install -r requirements.txt` in the terminal
3. Create a `.env` file with: `ANTHROPIC_API_KEY=your_key_here`
4. run `python app.py` in the terminal
5. Open link shown in the terminal

## How It Works (At a high level)

- **Retrieval:** TF-IDF vectorizer fit on all job texts at startup. At query time, the candidate text is augmented with current boost and required preference terms and compared against all jobs via cosine similarity. Top 100 by TF-IDF score are passed to the ranking stage.

- **Ranking:** Preference-weighted scoring on top of TF-IDF similarity. `+0.1` per boost term match, `-0.15` per penalize term match, `-999` if a require synonym group has no match. `require` is a list of synonym groupsand a job passes a requirement if it matches any term in the group (e.g. `[["bay area", "sf", "san jose"]]`). Top 3 results after sorting are returned to the user.

- **Feedback parsing:** Claude maps natural language feedback to structured `{ boost, penalize, require }` preference deltas with synonym expansion (abbreviations, geographic neighbors, role variants), merging with existing preferences across rounds so context accumulates.

## Tradeoffs & What I'd Improve

- Swap TF-IDF for semantic embeddings for better recall on semantically related but lexically different terms.
- When jobs exceed 100k+, add approximate nearest neighbor search (Pinecone, FAISS) instead of brute-force cosine similarity.
- Persist sessions to a database (SQLite maybe) for cross-refresh continuity.
- Add a proper candidate onboarding flow instead of raw JSON input.