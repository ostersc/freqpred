// forked from design-sync lib/source-kit.mjs — synth-entry mode's blanket
// `export * from <file>` (a) silently drops default-exported components
// (AssessmentCard, PriceTimeline — ES `export *` never re-exports `default`)
// and (b) bundles every .tsx under srcDir regardless of cfg.componentSrcMap
// exclusions, pulling data-fetching feature components (NavBar, Footer,
// PositionDetail, SignalDetail, AnalyzeButton — each calls useQuery/useMutation
// against freqpred's own API) and their deps (react-router-dom,
// @tanstack/react-query) into the shipped bundle even though they have no
// preview card. This fork filters those files out of the synthesized entry
// and adds a named re-export for any file with a detected PascalCase default
// export, so the exported set matches the intended 19-component scope.
//
// Non-storybook `package` adapter. Bundles dist/ when present (the authoritative
// component list comes from shipped .d.ts; with no dist it synthesizes an
// entry from src/ as a last resort) and opportunistically enriches each
// component from src/ — JSDoc and dir-derived group. Every enrichment miss
// degrades to the plain-dist behaviour.
//
// Discovery is heuristic-based; each heuristic has a `.design-sync/config.json`
// override (ASSUMPTION comments below name them) so repos that don't match the
// defaults write config, not code. `componentSrcMap` is the single override
// knob for component inclusion: non-null value = add/pin src path, null =
// exclude a .d.ts-exported internal.

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { Project, Node, ts } from 'ts-morph';
import { leadingJsdoc, readText, slash, walk } from '../../.ds-sync/lib/common.mjs';
import { resolveDistEntry } from '../../.ds-sync/lib/bundle.mjs';
import { exportedNames, isComponentName } from '../../.ds-sync/lib/dts.mjs';

const NON_IMPL_RX = /\.(stories|test|spec)\./;
const SRC_IMPL_RX = /\.(tsx|jsx)$/;
// Dir names that don't usefully group components — skip so the emitted path
// is `components/<group>/<Name>` not `components/components/<Name>`.
const GENERIC_DIR = new Set(['components', 'component', 'src', 'lib', 'ui', 'packages', 'react']);
const slug = (s) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'general';

// No .d.ts → scan src files for PascalCase value exports via ts-morph.
function deriveComponentsFromSrc(srcFiles) {
  const project = new Project({
    skipAddingFilesFromTsConfig: true,
    compilerOptions: { jsx: ts.JsxEmit.Preserve, allowJs: true, skipLibCheck: true },
  });
  const seen = new Set();
  for (const p of srcFiles) {
    if (NON_IMPL_RX.test(p) || !SRC_IMPL_RX.test(p)) continue;
    const sf = project.addSourceFileAtPathIfExists(p);
    if (!sf) continue;
    for (const [name, decls] of sf.getExportedDeclarations()) {
      // `export default function Button()` is keyed as 'default' — recover
      // the declared name from the function/class node.
      const real = name === 'default'
        ? decls.map((d) => d.getName?.()).find((n) => n && n !== 'default')
        : name;
      if (!real || !/^[A-Z][A-Za-z0-9]*$/.test(real)) continue;
      if (decls.some((d) => Node.isVariableDeclaration(d) || Node.isFunctionDeclaration(d) || Node.isClassDeclaration(d))) {
        seen.add(real);
      }
    }
  }
  return [...seen].sort().map((name) => ({ name, group: 'general' }));
}

// Lightweight regex probe for `export default function <Name>` / `export
// default <Name>` — good enough to decide whether a synthesized-entry re-export
// line is needed; full resolution still happens later via ts-morph.
const DEFAULT_EXPORT_RX = /export\s+default\s+function\s+([A-Z][A-Za-z0-9]*)\b|\bexport\s+default\s+([A-Z][A-Za-z0-9]*)\s*;?\s*$/m;

export async function resolvePackage(ctx) {
  const { PKG_DIR, pkgJson, ENTRY_OVERRIDE, PKG, OUT, cfg } = ctx;
  const srcMap = cfg.componentSrcMap ?? {};
  const excludedBasenames = new Set(Object.entries(srcMap).filter(([, v]) => v === null).map(([k]) => k));

  // ── 1. src/ discovery (best-effort; feeds enrichment + synth-entry fallback).
  // ASSUMPTION: source root is first of src/ | lib/ | components/. Override: cfg.srcDir.
  const srcRoot = [cfg.srcDir, 'src', 'lib', 'components']
    .map((d) => d && resolve(PKG_DIR, d))
    .find((d) => d && existsSync(d));
  const srcFiles = srcRoot ? walk(srcRoot, (n) => /\.(tsx|jsx|mdx?)$/.test(n)) : [];

  // ── 2. entry: dist if it exists, else synthesize from src/ (last resort).
  let entry = resolveDistEntry({ pkgDir: PKG_DIR, pkgJson, override: ENTRY_OVERRIDE, pkgName: PKG, soft: true });
  let synthEntry = false;
  if (!entry) {
    if (!srcRoot) {
      console.error(`[NO_DIST] ${PKG} has no built entry and no src/ to synthesize from — run its build.`);
      process.exit(1);
    }
    const comps = srcFiles
      .filter((p) => SRC_IMPL_RX.test(p) && !NON_IMPL_RX.test(p))
      .filter((p) => !excludedBasenames.has(basename(p).replace(/\.(tsx|jsx)$/, '')));
    const lines = [];
    for (const p of comps) {
      lines.push(`export * from ${JSON.stringify(p)};`);
      const m = DEFAULT_EXPORT_RX.exec(readText(p));
      const defaultName = m?.[1] ?? m?.[2];
      if (defaultName) lines.push(`export { default as ${defaultName} } from ${JSON.stringify(p)};`);
    }
    entry = join(OUT, '.pkg-entry.mjs');
    writeFileSync(entry, lines.join('\n') + '\n');
    synthEntry = true;
    console.error(
      `[NO_DIST] no built entry — synthesizing from ${comps.length} src files (run the package's build for best results)`,
    );
  }

  // ── 3. component list: from shipped .d.ts (authoritative when dist exists).
  // ASSUMPTION: components = PascalCase value exports in the .d.ts tree.
  // Override: cfg.componentSrcMap (non-null adds/pins, null excludes).
  const exported = exportedNames(PKG_DIR, pkgJson);
  const names = new Set([...exported].filter(isComponentName));
  for (const [k, v] of Object.entries(srcMap)) {
    if (v === null) { names.delete(k); continue; }
    // Names reach `<script>` blocks in the emitted HTML — reject anything
    // that isn't a plain PascalCase identifier.
    if (!/^[A-Z][A-Za-z0-9]*$/.test(k)) {
      console.error(`[CONFIG] componentSrcMap: "${k}" is not a valid component name (PascalCase identifiers only)`);
      continue;
    }
    names.add(k);
  }
  let components = [...names].sort().map((name) => ({ name, group: 'general' }));
  if (!components.length && synthEntry) {
    components = deriveComponentsFromSrc(srcFiles).filter((c) => srcMap[c.name] !== null);
  }
  if (!components.length) {
    if (cfg.cssEntry || existsSync(join(PKG_DIR, 'styles.css'))) {
      console.error('[ZERO_MATCH] no component exports — treating as tokens-only DS');
      return { shape: 'package', entry, components: [], tokensOnly: true };
    }
    console.error(`[ZERO_MATCH] no PascalCase exports in ${PKG} and no styles — nothing to sync`);
    process.exit(1);
  }

  // ── 4. src/ enrichment per component. Every miss degrades to plain-dist.
  if (srcRoot) {
    for (const c of components) {
      // Pinned via config → skip fuzzy-find entirely.
      let hit = typeof srcMap[c.name] === 'string' ? slash(resolve(PKG_DIR, srcMap[c.name])) : null;
      if (!hit) {
        // ASSUMPTION: <Name>.tsx | <name>/<name>.tsx | <Name>/index.tsx |
        // <kebab-name>.tsx, case-insensitive; dir-match ranks above
        // bare-file match, then prefer one that actually exports `c.name`.
        // Override: cfg.componentSrcMap.
        const kebab = c.name.replace(/([a-z0-9])([A-Z])/g, '$1-$2');
        const nameRx = new RegExp(
          `(?:^|/)(?:${c.name}/(?:index|${c.name})\\.(tsx|jsx)|(?:${c.name}|${kebab})\\.(tsx|jsx))$`,
          'i',
        );
        const hits = srcFiles
          .filter((p) => nameRx.test(p) && !NON_IMPL_RX.test(p))
          .sort(
            (a, b) =>
              (b.toLowerCase().includes(`/${c.name.toLowerCase()}/`) ? 1 : 0) -
              (a.toLowerCase().includes(`/${c.name.toLowerCase()}/`) ? 1 : 0),
          );
        const exportRx = new RegExp(`export\\s+(?:default\\s+)?(?:const|let|var|function|class)\\s+${c.name}\\b`);
        hit = hits.find((p) => exportRx.test(readText(p))) ?? hits[0];
      }
      if (!hit || !existsSync(hit)) continue;
      c.srcPath = hit;
      c.doc = leadingJsdoc(readText(hit), c.name) || undefined;
      // group = last src/ path segment that isn't the component's own dir or
      // a generic container name — else JSDoc @category — else 'general'.
      c.group = slug(
        slash(relative(srcRoot, dirname(hit)))
          .split('/')
          .filter((s) => s && s.toLowerCase() !== c.name.toLowerCase() && !GENERIC_DIR.has(s.toLowerCase()))
          .at(-1)
        || (c.doc && /@category\s+(\S+)/.exec(c.doc)?.[1])
        || 'general',
      );
    }
  }

  console.error(
    `  package: ${components.length} components` +
      (srcRoot ? ` (${components.filter((c) => c.srcPath).length} src-matched)` : ' (no src/ — dist-only)'),
  );
  return { shape: 'package', entry, components, synthEntry, exported };
}
