## freqpred-dashboard conventions

This is freqpred's own trading-dashboard UI kit — a dark, dense "terminal"
aesthetic, not a general-purpose product design system. Build with the real
components below; don't invent generic ones.

### No provider needed for styling — but wrap navigable compositions in a Router

Every color/spacing value comes from CSS custom properties on `:root` in
`styles.css` — nothing needs a theme provider. The one exception: `AssessmentCard`
renders a react-router-dom `<Link>` when its `llm_query_id` prop is set. If you
compose it (or anything using in-app navigation) into a page, wrap the tree in
a `<MemoryRouter>` (or `<BrowserRouter>` in a real app) — un-wrapped, `<Link>`
throws.

### Styling idiom: hand-authored CSS classes + CSS custom properties

This is **not** a Tailwind/utility-class system and **not** a props-only
styling API. Components apply a small set of hand-authored class names
(`className="badge pos"`, `.panel`, `.stat`, `.seg`, `.mini-stat`,
`.labeled-field`, `.input`, `.select`, `.range-slider`, `.error-banner`,
`.warn-banner`, `.spinner-ring`, `.mono`, `.num`) and everything those classes
draw from lives in `:root` custom properties — reuse these, don't invent new
hex values:

| Token | Use |
|---|---|
| `--bg-0` … `--bg-3` | surface layers, darkest to lightest |
| `--fg-0` … `--fg-3` | text, brightest to dimmest |
| `--line`, `--line-soft`, `--line-strong` | borders/dividers |
| `--accent`, `--accent-soft`, `--accent-line` | primary brand accent (violet-blue) |
| `--pos` / `--pos-soft` | positive/gain (green) |
| `--neg` / `--neg-soft` | negative/loss (red) |
| `--warn` / `--warn-soft` | caution (amber) |
| `--info` / `--info-soft` | neutral-informational (blue) |
| `--r-sm` `--r-md` `--r-lg` | border-radius scale (6/10/14px) |
| `--f-sans` (Inter Tight) / `--f-mono` (JetBrains Mono) | typefaces — numbers and tickers are almost always `--f-mono` via the `.mono`/`.num` classes |

`Badge` is the canonical example of the semantic-kind pattern used throughout:
`kind` is one of `pos | neg | warn | info | accent | muted`, each mapping to a
`--<kind>` / `--<kind>-soft` pair. Follow that same six-kind vocabulary
anywhere you need a status color — don't introduce a seventh.

### Where the truth lives

Read `styles.css` (it `@import`s the token sheet and `_ds_bundle.css`) before
styling anything by hand. Each component's real prop contract is in its
`<Name>.d.ts` next to its preview card — trust that over guessing, since this
repo has no Storybook and props were hand-transcribed from source.

### Composition example

```tsx
import { Panel, Stat, Badge, Sparkline } from 'freqpred-dashboard'

<Panel title="Open positions" action={<Badge kind="accent">Live</Badge>}>
  <Stat
    label="Open P&L"
    value="+$482.10"
    delta="+2.1%"
    deltaKind="pos"
    spark={<Sparkline data={[50, 52, 49, 55, 58, 56, 61, 64]} />}
  />
</Panel>
```

Layout glue (grids, flex containers) that isn't one of these components should
use inline `style` with the same CSS custom properties — that's how every
component here does it internally (no CSS-in-JS library, no utility classes).
