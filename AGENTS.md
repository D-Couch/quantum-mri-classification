# Development instructions

- Optimize generalization on held-out patients; choose models by validation macro-F1.
- Preserve dataset class definitions and provenance. Split by patient/group where available, deduplicate before splitting, fit preprocessing only on training data, and augment training data only.
- Never tune on the final test set. Never commit MRI data, credentials, or generated artifacts.
- Store experiment metadata in SQLite and bulky files under the ignored local `artifacts/` directory.
- Keep dashboard code separate from datasets, models, training, metrics, and persistence.
- Record seeds, dataset version, configuration, environment, revision, and measured outcomes. Label synthetic and incomplete runs clearly.
- Use `python -m pytest` for tests and `streamlit run app.py` to launch the local dashboard. Do not claim hardware or real-data results unless executed.
- Keep dependencies in `pyproject.toml`; use a virtual environment. Review `docs/REQUIREMENTS.md` and `docs/ROADMAP.md` before changes and update the roadmap when milestones move.
- Make small, reviewable commits; do not push unless authenticated and repository access is verified.
