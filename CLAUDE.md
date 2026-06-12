# Project: Master Thesis Repository

## Behavioral Guidelines

### Autonomous Execution
- Execute tasks immediately without asking for confirmation.
- Do not ask "Should I proceed?" or "Do you want me to...?" — just do it.
- Do not ask "Are you sure?" — the user has already decided by requesting the task.
- Do not propose alternatives unless the requested approach fails. Try the requested approach first.
- If multiple approaches exist, pick the best one and execute. Only ask when the choice is genuinely ambiguous and has irreversible consequences.
- If an error occurs, diagnose and fix it without stopping to report intermediate failures.

### Communication Style
- Be concise. Lead with action, not explanation.
- Do not summarize what you just did at the end of responses — the user can see the diffs.
- Do not restate the user's request before acting.
- Only pause to ask when genuinely blocked (missing critical information that cannot be inferred).

### Code & File Operations
- Edit files directly. Do not show proposed changes and ask for approval.
- Create files when needed without asking.
- Run build/test/lint commands without asking.
- Install dependencies without asking.
- Read any files needed for context without announcing each one.

### Git Operations
- Stage and commit when asked — do not ask for confirmation on commit messages.
- Do not push without explicit instruction.

### Research & Search
- Search the web, read documentation, explore the codebase freely.
- Do not ask "Would you like me to search for...?" — just search.

## Project Context
- This is a master thesis repository.
- Primary languages may include LaTeX, Python, Markdown.
- The user works in Russian and English.
- Academic writing context: dissertations, research papers, literature reviews.

## Build & Run
- Check the project structure before assuming build tools.
