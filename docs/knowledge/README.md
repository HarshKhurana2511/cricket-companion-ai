# Knowledge Base (Seed Corpus)

This folder contains the seed knowledge corpus used by the **Basic Cricket Chatbot** (Phase `2.5`).

## Goals

- Short, accurate, retrieval-friendly cricket explanations (rules, formats, tactics, glossary).
- Written to be chunked and embedded later (Phase `2.5.2`).
- Stable sources for citations (file path + heading).

## Authoring conventions

- Prefer many small sections over long prose.
- Use clear headings (`##`, `###`) so chunking can align with topics.
- Keep definitions explicit (don’t assume prior knowledge).
- When a concept has multiple definitions in practice, state the project’s convention and note alternatives.

## Structure

- `rules/`: rules, outcomes, common edge cases
- `formats/`: format differences (Test/ODI/T20)
- `tactics/`: strategy concepts by phase/role
- `stats/`: definitions of key metrics (economy, strike rate, dots, boundaries)
- `glossary.md`: quick definitions and cross-references

