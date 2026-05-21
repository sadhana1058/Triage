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

This command will compare the predictions in `support_tickets/output.csv` with the ground truth labels and display evaluation metrics.

## File Structure

- `code/`: Contains the main codebase.
  - `main.py`: The entry point of the application.
  - `agent.py`: Implements the agentic loop and orchestrates the pipeline.
  - `retriever.py`: Handles the retrieval of relevant support documents.
  - `guardrails.py`: Implements pre-flight and post-generation checks.
  - `tracer.py`: Provides tracing and logging functionality.
  - `evaluator.py`: Evaluates the model's performance against ground truth.
- `data/`: Contains the support corpus for each domain.
- `support_tickets/`: Contains the input CSV file and the generated output.
- `traces/`: Contains the trace files for each run.
- `.env`: Environment configuration file.
- `requirements.txt`: Lists the required Python dependencies.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).