# Dashboard UI conventions

## Text color on colored backgrounds — always use CSS variables, never hardcoded colors

When rendering text inside cells or elements that have a dynamically-colored background (e.g. heatmap cells, badges, status indicators), use `var(--fg-0)` for the text color instead of hardcoded values like `#4ade80` or `#f87171`.

**Why:** The dashboard supports both dark mode and light mode. Hardcoded light colors (e.g. bright green `#4ade80`) are invisible on light-mode backgrounds of the same hue — green text on a light-green cell. CSS variables like `var(--fg-0)` resolve to white in dark mode and near-black in light mode, giving correct contrast in both themes automatically.

**Rule:** If text sits on top of a background whose color is determined at runtime (e.g. from a delta value or status), use `var(--fg-0)`. Direction or severity is already communicated by the background color and the +/− sign; repeating the color in the text is redundant and causes legibility problems.

```tsx
// Wrong — green text on green cell = invisible in light mode
<div style={{ color: delta > 0 ? '#4ade80' : '#f87171' }}>
  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
</div>

// Correct — CSS variable adapts to theme automatically
<div style={{ color: 'var(--fg-0)', opacity: 0.8 }}>
  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
</div>
```

This applies to: heatmap cell content, badge labels, inline stat values, and any other text rendered on a programmatically-colored surface.
