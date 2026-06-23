# tsdFlow viewport-UX sketches (2026-06-23)

Exploring a **unified viewports control** that both reveals viewport windows and sets
each display's per-viewport membership — replacing today's View-menu + 8-checkbox-row split.
Open the SVGs in a browser/preview.

| File | Idea | Reveal viewports | Per-display masking |
|------|------|------------------|---------------------|
| `proposal-1-chips-on-node.svg` | **Chips on the node** | "+" tab in the viewport tab strip | numbered chips on each display node (filled = renders there) |
| `proposal-2-viewports-bar.svg` | **Unified Viewports bar** | eye on each cell of one docked strip | same strip's cells fill for the *selected* display |
| `proposal-3-hybrid-chips-plus-reveal.svg` | **Hybrid: chips + slim rail** | slim vertical viewport rail (click = show/hide window) | chips on the node (as P1) |

**Trade-offs**
- **P1 (chips on node):** masking is visible for *every* display at once, no selection needed; matches "chips in the object box". Reveal is a separate small affordance.
- **P2 (one bar):** single control for both jobs; but masking is selection-dependent (only shows the selected display's membership) and the bar overloads two meanings (eye=window, fill=membership) per cell.
- **P3 (hybrid):** cleanest separation — rail owns *where windows are*, chips own *what each display shows*; two tiny always-visible controls, no menus, no Inspector grid. Closest to "unified yet lean".

Auto-layout (#3) is handled separately (layered DAG placement on add + a "Clean Up Layout" button); not sketched here since it's a known pattern rather than a visual-design choice.
