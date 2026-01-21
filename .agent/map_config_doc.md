# map_config_doc.md

everything-claude-code/
  |-- agents/                  # Specialized subagents for delegation
  |-- planner.md               # Feature implementation planning
      architect.md             # System design decisions
      tdd-guide.md             # Test-driven development
      code-reviewer.md               # Quality and security review
      security-reviewer.md           # Vulnerability analysis
      build-error-resolver.md
      e2e-runner.md
      refactor-cleaner.md      # Dead code cleanup
  |--doc-updater.md            # Documentation sync
      skills/                  # Workflow definitions and domain knowledge
|-- coding-standards.md
# Language best practices
|-- backend-patterns.md
# API, database, caching patterns
frontend-patterns.md
# Playwright E2E testing
# React, Next.js patterns
|-- project-guidelines-example.md
# Example project-specific skill
|-- tdd-workflow/
security-review/
# TDD methodology
# Security checklist
|-- clickhouse-io.md
# ClickHouse analytics
commands/
# Slash commands for quick execution
Implementation planning #/plan
|-- tdd.md
#/tdd Test-driven development
plan.md
e2e.md
#/e2e E2E test generation
code-review.md
#/code-review Quality review
build-fix.md
#/build-fix Fix build errors
refactor-clean.md
#/refactor-clean Dead code removal
test-coverage.md
#/test-coverage Coverage analysis
update-codemaps.md
# /update-codemaps Refresh docs
|-- update-docs.md
# /update-docs Sync documentation
#Always-follow guidelines
rules/
|-- security.md
|-- coding-style.md
|-- testing.md
git-workflow.md
|-- agents.md
performance.md
patterns.md
hooks.md
# Mandatory security checks
# Immutability, file organization
#TDD, 80% coverage requirement
# Commit format, PR process
#When to delegate to subagents
# Model selection, context management
# API response formats, hooks
#Hook documentation
hooks/
#Trigger-based automations
|-- hooks.json
# PreToolUse, PostToolUse, Stop hooks
mcp-configs/
# MCP server configurations
|-- mcp-servers.json
# GitHub, Supabase, Vercel, Railway, etc.
plugins/
|-- README.md
# Plugin ecosystem documentation
# Plugins, marketplaces, skills guide
|-- examples/
# Example configurations
|-- CLAUDE.md
# Example project-level config
|-- user-CLAUDE.md
# Example user-level config (~/.claude/CLAUDE.md)
|-- statusline.json
# Custom status line config
