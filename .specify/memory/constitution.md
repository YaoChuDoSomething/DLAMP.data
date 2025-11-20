# DLAMP.data Constitution

<!--
Version change: 0.0.0 → 1.0.0
List of modified principles:
  - "不可變更性 (Immutability)"
  - "職責單一化 (Single Responsibility)"
  - "設定檔驅動 (Configuration-Driven)"
  - "數據接口標準化 (Standardized Data Interface)"
  - "可擴展性 (Extensibility)"
Added sections: Goals
Removed sections: Core Principles, Additional Constraints, Development Workflow
Templates requiring updates:
  - .specify/templates/plan-template.md ⚠ pending
  - .specify/templates/spec-template.md ⚠ pending
  - .specify/templates/tasks-template.md ⚠ pending
  - .specify/templates/commands/speckit.analyze.toml ⚠ pending
  - .specify/templates/commands/speckit.checklist.toml ⚠ pending
  - .specify/templates/commands/speckit.clarify.toml ⚠ pending
  - .specify/templates/commands/speckit.constitution.toml ✅ updated
  - .specify/templates/commands/speckit.implement.toml ⚠ pending
  - .specify/templates/commands/speckit.plan.toml ⚠ pending
  - .specify/templates/commands/speckit.specify.toml ⚠ pending
  - .specify/templates/commands/speckit.tasks.toml ⚠ pending
  - .specify/templates/commands/speckit.taskstoissues.toml ⚠ pending
Follow-up TODOs: None
-->

### **Phase 1: Constitution (基本構想 / 原則)**

此階段確立專案的核心目標與不可動搖的原則。

**原則 (Principles):**

1.  **不可變更性 (Immutability):** 嚴格遵守不修改 `@GEMINI.md`、`@src/preproc/` 及 `@src/registry/` 目錄下任何檔案的規定。
2.  **職責單一化 (Single Responsibility):** `@src/opflows/` 目錄下的每個 Python 模組都應有明確且單一的職責（如下載、內插、運算、流程管理）。
3.  **設定檔驅動 (Configuration-Driven):** 所有流程的行為，如檔案路徑、時間範圍、變數名稱等，都應由 `@config/opflows/` 目錄下的 YAML 檔案控制，而非寫死在程式碼中。
4.  **數據接口標準化 (Standardized Data Interface):** 所有模組間的資料交換應以 NetCDF4 格式為標準。這能確保全球模式與資料來源 (`earth2studio.data.fetch_data` 或 GRIB 檔案) 的解耦。
5.  **可擴展性 (Extensibility):** 整體架構應易於擴展，未來要新增其他模式（如 ICON、IFS）或新的工作流程時，應只需新增對應的 operator 和修改設定檔即可。

**目標 (Goals):**

1.  **重構現有流程:** 將「區域模式資料前處理流程」依照新的模組化架構進行重構。
2.  **建構單向耦合流程:** 實現一個由 SFNO 預報驅動 RWRF 的單向降尺度工作流程。
3.  **建構雙向回饋框架:** 為 `FeedbackUpdater` 建立基礎，使其能夠將高解析度區域資料更新回全球模式的初始場。
4.  **建立測試機制:** 在 `@src/opflows/tests/` 中建立單元測試與整合測試，確保每個模組的功能正確性及流程的穩定性。

## Governance
Constitution supersedes all other practices; Amendments require documentation, approval, migration plan. All PRs/reviews must verify compliance; Complexity must be justified.

**Version**: 1.0.0 | **Ratified**: 2025-11-21 | **Last Amended**: 2025-11-21