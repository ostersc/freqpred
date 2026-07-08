# freqpred-dashboard design-sync notes

## What this syncs

`freqpred/dashboard/ui` is the freqpred app itself (a Vite SPA), not a
published component library — there is no `dist/` library build, no
Storybook, and `src/components/` mixes reusable presentational pieces with
app-specific data-fetching feature components. This sync deliberately scopes
to the **19 presentational components** (`ui.tsx`'s 16 primitives +
`AssessmentCard`, `PriceTimeline`, `DocLinkItem`) and excludes `NavBar`,
`Footer`, `PositionDetail`, `SignalDetail`/`SelectedSignalPanel`, and
`AnalyzeButton` — each of those calls `useQuery`/`useMutation` against
freqpred's own API and isn't reusable design-system material.

## Synth-entry mode (no dist)

No library build exists, so this runs in the converter's "no dist" synth-entry
fallback (`cfg.srcDir: "src/components"`), which scans `.tsx` files with
ts-morph instead of reading a shipped `.d.ts` tree. Two consequences:

- **Prop contracts are hand-written**, not extracted (`cfg.dtsPropsFor` for
  all 19 components) — the auto-extraction from a real dist `.d.ts` tree
  doesn't exist in this mode, so it stubs `[key: string]: unknown` otherwise.
  If a component's real prop signature changes in `src/components/`, update
  the matching `dtsPropsFor` entry by hand — nothing re-derives it.
- **`.design-sync/overrides/source-kit.mjs` is forked** (declared in
  `cfg.libOverrides`) to fix two synth-entry gaps: (a) plain `export *` never
  re-exports a file's `default` export, which silently dropped
  `AssessmentCard` and `PriceTimeline` from the bundle; (b) the unforked
  synth-entry step bundles every `.tsx` under `srcDir` regardless of
  `componentSrcMap` null-exclusions, so the excluded data-fetching components
  (and their deps — react-router-dom, @tanstack/react-query) were still
  getting pulled into `_ds_bundle.js` even with no preview card. If the repo
  ever ships a real library build (`vite build --lib` or similar) with an
  `exports`/`module` entry in `package.json`, switch off synth-entry
  entirely — drop `srcDir`/`componentSrcMap` exclusions in favor of a curated
  entry file, and this fork becomes unnecessary.

## Fonts

`Inter Tight` and `JetBrains Mono` are loaded at runtime via a Google Fonts
`<link>` in `freqpred/dashboard/ui/index.html`, not shipped as files. Per user
sign-off, self-hosted copies were fetched from `fonts.gstatic.com` (public,
license-permitting) into `.design-sync/assets/fonts/` and wired via
`cfg.extraFonts`. Both are variable fonts — Google serves the identical woff2
file for every requested static weight, so `fonts.css` declares one
`@font-face` per family with a `font-weight` range rather than five duplicate
blocks. If the app ever changes its Google Fonts request (different weights
or a new family), re-fetch and update `.design-sync/assets/fonts/`.

## Router provider

`AssessmentCard` conditionally renders a react-router-dom `<Link>` (the "Open
LLM audit" link, only when `llm_query_id` is set). `cfg.provider` wraps every
preview in a `MemoryRouter` and `cfg.extraEntries` adds `react-router-dom` to
the bundle so that link renders instead of throwing. Harmless for every other
component (none read router context).

## Known render warns (confirmed benign, don't re-chase on re-sync)

- `Donut` and `Sparkline` trip `[RENDER_THIN]` ("mounts have no text and
  paint nothing") on every build — both are pure-SVG components with no text
  nodes, so the text-based heuristic false-positives. Confirmed against the
  actual screenshots (`_screenshots/review/general__Donut.png` and
  `general__Sparkline.png`) — both paint correctly.

## Preview limitations

- `DocLinkItem`'s expanded-detail state (click the "i" toggle) is
  interaction-only and can't render statically — only the collapsed list is
  shown. Same for any other click-to-reveal state in these components.

## Re-sync risks

- The `dtsPropsFor` bodies in `.design-sync/config.json` are a hand-maintained
  mirror of each component's real TS prop signature (and, for `AssessmentCard`
  / `PriceTimeline` / `DocLinkItem`, a trimmed-down structural shape of the
  freqpred API types they consume — `SignalAssessmentOut`, `SignalOut`,
  `DocumentLinkOut` in `freqpred/dashboard/ui/src/api/types.ts` — rather than
  the full DTOs). If those source props or API types change, the emitted
  `.d.ts` will silently go stale until someone updates `dtsPropsFor` by hand.
- The `.design-sync/overrides/source-kit.mjs` fork should be diffed against
  the bundled `lib/source-kit.mjs` on every re-sync in case the upstream skill
  adds a config knob that makes the fork unnecessary.
- Bundle size is ~1.3MB, mostly `recharts` (pulled in by `PriceTimeline`) —
  expected, not a regression to chase.
