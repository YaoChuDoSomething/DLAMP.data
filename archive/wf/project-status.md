---
description: Output the Project Status
---

# Project Status Update Workflow

This workflow ensures project documentation and diagrams are strictly aligned with reality.

1. **Synthesize Current State**
   - Use `sequential-thinking` to analyze the actual state of the code and tasks.
   - Compare with `task.md` and current documents in `.agent/pmdoc/`.

2. **Alignment Loop (Sequential Thinking)**
   - **Goal**: Ensure alignment between Actual Status and Documentation/Charts.
   - **Step 2.1 Identify Misalignments**: List discrepancies in:
     - Gantt Charts (Timeline)
     - Kanban Boards (Task status)
     - Milestones
     - Project Asset Tables
     - Architecture Diagrams
     - Class Diagrams
   - **Step 2.2 Execute Modification**:
     - Update Markdown tables.
     - Update Mermaid diagrams.
     - **Tip**: Use `context7` tool to query Mermaid documentation or examples if complex syntax is needed.
   - **Step 2.3 Verify**:
     - Check that Mermaid code is valid and renders a preview.
     - Confirm the info matches step 1.

3. **Update Documents**
   - Save updated charts/tables to `.agent/pmdoc/status.md` (or specific files like `architecture.md`).

4. **Output Status**
   - Present a concise summary to the User, confirming that diagrams are now aligned.
