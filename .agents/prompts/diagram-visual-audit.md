# Visual E2E audit — thesis diagram (Graphviz)

You are reviewing a **rendered PNG** of a thesis diagram (master's dissertation, Russian labels, SAT color notation).

## Input

- PNG path is provided in the task.
- Optional: `layout_report.json` geometry pre-check from `scripts/check_svg_layout.py`.

## Checklist (mark each PASS or FAIL)

1. **legend_visible** — SAT legend is fully visible, not empty, not covered by a white box.
2. **no_overlaps** — No overlapping boxes, labels, or unreadable edge crossings.
3. **cluster_bounds** — Cluster frames (bi-encoder, LLM, system boundary on C4) fully contain their nodes.
4. **flow_direction** — Main spine and branch chains read left-to-right; branches are not accidental vertical stacks.
5. **scale_balance** — No huge empty bands; text readable at ~1200px width.
6. **sat_colors** — Border/fill colors match semantic roles (implemented=green, experimental=teal, approach=blue, assumption=amber dashed, concept=purple dashed).

## Output format (strict)

```markdown
## {diagram_stem}

**Verdict:** PASS | FAIL

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | legend_visible | PASS/FAIL | ... |
...

### Blocking issues
- (list or "none")

### Suggested .dot fixes
- (concrete: node ids, rankdir, rank=same, nodesep, legend node type)
```

Be strict: if anything hurts readability in a thesis PDF/Obsidian embed, mark FAIL.
