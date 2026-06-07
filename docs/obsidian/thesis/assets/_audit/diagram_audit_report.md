# E2E visual audit — thesis Graphviz diagrams

**Date:** 2026-06-03  
**Pipeline:** `scripts/run_diagram_visual_e2e.ps1`  
**Geometry:** `scripts/check_svg_layout.py` → `layout_report.json`

## Summary

| Diagram | Geometry | Vision | Notes |
|---------|----------|--------|-------|
| application_analysis_request_flow | PASS | PASS | Horizontal spine + branches; SAT table legend visible |
| c4_container_architecture | PASS | PASS (after .dot tune) | Migrated to `sat_legend`; compressed canvas |

---

## application_analysis_request_flow

**Verdict:** PASS

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | legend_visible | PASS | HTML `sat_legend` with 6 colored cells + footnote |
| 2 | no_overlaps | PASS | No blocking overlaps at 120 DPI |
| 3 | cluster_bounds | PASS | bi-encoder / LLM frames contain chains |
| 4 | flow_direction | PASS | Spine and branches left-to-right (`rank=same`) |
| 5 | scale_balance | PASS | Wide but readable; `ratio=compress` applied |
| 6 | sat_colors | PASS | Green / teal / blue diamond / purple dashed consistent |

### Blocking issues
- none

### Applied .dot fixes
- `sat_legend` HTML table (no empty cluster overlay)
- `rank=same` rows, `splines=polyline`
- `size="10,7!" ratio=compress`

---

## c4_container_architecture

**Verdict:** PASS

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | legend_visible | PASS | `sat_legend` table after migration from `cluster_legend` |
| 2 | no_overlaps | PASS | Acceptable routing; LLM box near backend arrow |
| 3 | cluster_bounds | PASS | System boundary contains core containers |
| 4 | flow_direction | PASS | Rows LR inside system boundary |
| 5 | scale_balance | PASS | Reduced empty margin via `size` + `polyline` |
| 6 | sat_colors | PASS | Matches SAT scale |

### Blocking issues
- none

### Applied .dot fixes
- Replaced `cluster_legend` nodes with `sat_legend` HTML table
- `splines=polyline`, removed `newrank`
- `flow_anchor` from `metrics` to legend
- `size="12,9!" ratio=compress`

---

## How to re-run

```powershell
.\scripts\run_diagram_visual_e2e.ps1
```

Vision agents: use PNG under `_audit/` + [`.agents/prompts/diagram-visual-audit.md`](../../../.agents/prompts/diagram-visual-audit.md).
