---
description: Quality Gate Check
---

# Quality Gate Workflow

Automated quality assurance.

1. **Initiate Check**
   - Scope: Modified files or full project.

2. **Linting & Formatting**
   - Run linters (ruff, black, etc.).
   - Auto-fix where possible.

3. **Docs Sync**
   - Check `README.md` vs Code.
   - Check `task.md`.

4. **Tests**
   - Run unit tests.

5. **Report**
   - Pass/Fail status.
