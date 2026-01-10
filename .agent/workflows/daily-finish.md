---
description: Finish the Daily Progress
---

# Daily Finish Workflow

Routine to finalize and save work, prioritized by specification level.

1. **Determine Action (Highest Spec First)**
   - Use `sequential-thinking` to evaluate the state.
   - **Level 1: Pull Request (High Spec)**
     - Is the feature complete? Is it ready for review?
     - Action: `create_pull_request` or `update_pull_request` (merge if authorized).
   - **Level 2: Push (Medium Spec)**
     - Are there local commits? Is the work safe to share?
     - Action: `git push`.
   - **Level 3: Commit (Basic Spec)**
     - Are there uncommitted changes?
     - Action: `git add .` and `git commit -m "..."`.

2. **Summarize Work**
   - detailed summary of what was accomplished.

3. **Execute Action**
   - Perform the action determined in Step 1.

4. **Update Documentation**
   - Update `task.md`.
   - Trigger `project-status` workflow if significant changes occurred.