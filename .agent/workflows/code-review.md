---
description: Full Project Code Review
---

# Full Project Code Review Workflow

This workflow guides you through a comprehensive code review of the project.

1. **Initialize Thinking Process**
   - Use `sequential-thinking` to structure your review.
   - Goal: Understand the recent changes and the overall health of the project.

2. **Retrieve Recent Project Activity**
   - Use `remote-github` (e.g., `list_pull_requests`, `list_commits`) or `run_command` (`git log`) to see recent changes.

3. **Analyze Codebase Structure**
   - Use `list_dir` and `view_file_outline` to refresh understanding.

4. **Deep Dive & Review**
   - Read content of modified files.
   - Check for: Code style, bugs, security, and `@GEMINI.md` compliance.

5. **Generate Review Report**
   - Compile findings into a markdown report.
   - **Target Path**: `.agent/review/code_review_YYYYMMDD.md`
   - Include actionable items and severity levels.

6. **Finalize**
   - Summary of the review in the chat.
