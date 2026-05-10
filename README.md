# Triage
hackerrank-orchestrate-may26-main/
├── code/
│   ├── main.py          # CLI entry: `python main.py run` / `python main.py eval`
│   ├── agent.py         # Agentic loop (orchestrates all layers)
│   ├── retriever.py     # ChromaDB + MiniLM embeddings + cross-encoder re-rank
│   ├── guardrails.py    # Pre-flight + post-retrieval + post-generation checks
│   ├── tracer.py        # Per-ticket JSONL trace logging
│   ├── evaluator.py     # RAGAS + sklearn metrics (status F1, request_type F1, etc.)
│   └── README.md        # Install + run instructions
├── support_tickets/
│   ├── support_tickets.csv        # 56 input tickets (issue, subject, company)
│   ├── sample_support_tickets.csv # Labeled examples for eval/dev
│   └── output.csv                 # Agent writes predictions here
├── data/
│   ├── claude/          # Claude API, billing, privacy docs
│   ├── hackerrank/      # Assessments, interviews, SkillUp, integrations docs
│   └── visa/            # Travel, fraud, dispute, cheques docs
├── chroma_db/           # Auto-created: persisted ChromaDB vector store
├── traces/              # Auto-created: one .jsonl file per run
├── .env                 # GEMINI_API_KEY (gitignored)
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── problem_statement.md
└── evalutation_criteria.md