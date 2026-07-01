// REPL driver for the freqpred dashboard (React SPA served by Vite).
// Stand-in for `chromium-cli`, which isn't installed in this environment.
// Run under Node with Playwright installed in this directory (see SKILL.md).
// Designed for agents: run it directly, or wrap in tmux and send-keys commands.
import { chromium } from 'playwright';
import * as readline from 'node:readline';
import * as fs from 'node:fs';
import * as path from 'node:path';

const SHOT_DIR = process.env.SCREENSHOT_DIR || path.join(import.meta.dirname, 'screenshots');
fs.mkdirSync(SHOT_DIR, { recursive: true });

let browser = null;
let page = null;
const consoleErrors = [];

const COMMANDS = {
  async launch() {
    if (browser) return console.log('already launched');
    browser = await chromium.launch();
    page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    page.on('pageerror', (e) => consoleErrors.push(String(e)));
    page.on('console', (msg) => { if (msg.type() === 'error') consoleErrors.push(msg.text()); });
    console.log('launched');
  },

  async nav(url) {
    if (!page) return console.log('ERROR: launch first');
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30_000 });
    console.log('nav ->', url);
  },

  async 'wait-for'(sel) {
    if (!page) return console.log('ERROR: launch first');
    try { await page.waitForSelector(sel, { timeout: 15_000 }); console.log('found:', sel); }
    catch { console.log('TIMEOUT:', sel); }
  },

  async click(sel) {
    if (!page) return console.log('ERROR: launch first');
    try { await page.click(sel, { timeout: 10_000 }); console.log('click', sel, '-> OK'); }
    catch (e) { console.log('click', sel, '-> ERROR:', e.message.split('\n')[0]); }
  },

  async 'click-text'(text) {
    if (!page) return console.log('ERROR: launch first');
    try { await page.getByText(text, { exact: false }).first().click({ timeout: 10_000 }); console.log('click-text', JSON.stringify(text), '-> OK'); }
    catch (e) { console.log('click-text', JSON.stringify(text), '-> ERROR:', e.message.split('\n')[0]); }
  },

  async fill(args) {
    if (!page) return console.log('ERROR: launch first');
    const sp = args.indexOf(' ');
    const sel = sp === -1 ? args : args.slice(0, sp);
    const text = sp === -1 ? '' : args.slice(sp + 1);
    await page.fill(sel, text);
    console.log('fill', sel, '<-', JSON.stringify(text));
  },

  async press(key) { if (page) await page.keyboard.press(key); console.log('press', key); },

  async sleep(ms) { await new Promise((r) => setTimeout(r, Number(ms) || 500)); console.log('slept', ms + 'ms'); },

  async screenshot(name) {
    if (!page) return console.log('ERROR: launch first');
    const f = path.join(SHOT_DIR, (name || `ss-${Date.now()}`) + '.png');
    await page.screenshot({ path: f, fullPage: true });
    console.log('screenshot:', f);
  },

  async 'screenshot-element'(args) {
    if (!page) return console.log('ERROR: launch first');
    const sp = args.indexOf(' ');
    const sel = sp === -1 ? args : args.slice(0, sp);
    const name = sp === -1 ? undefined : args.slice(sp + 1);
    const f = path.join(SHOT_DIR, (name || `ss-${Date.now()}`) + '.png');
    await page.locator(sel).first().screenshot({ path: f });
    console.log('screenshot:', f);
  },

  async eval(expr) {
    if (!page) return console.log('ERROR: launch first');
    try { console.log(JSON.stringify(await page.evaluate(expr))); }
    catch (e) { console.log('ERROR:', e.message.split('\n')[0]); }
  },

  async text(sel) {
    if (!page) return console.log('ERROR: launch first');
    console.log(await page.evaluate(
      (s) => (s ? document.querySelector(s) : document.body)?.innerText ?? '(null)',
      sel || null,
    ));
  },

  async console(flag) {
    if (flag === '--errors' || !flag) {
      console.log(consoleErrors.length ? JSON.stringify(consoleErrors) : '(no errors)');
    }
  },

  async quit() { if (browser) await browser.close().catch(() => {}); browser = null; page = null; },
  help() { console.log('commands:', Object.keys(COMMANDS).join(', ')); },
};

const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: 'driver> ' });

// Piped input (heredoc) delivers every line in the same tick, so readline's
// 'line' events fire well before an async handler for a prior line (e.g.
// `launch`, which takes hundreds of ms) has resolved. Without an explicit
// queue, `nav`/`wait-for` right after `launch` see page still null and fail
// with a bogus "launch first". Interactive/tmux send-keys input never hits
// this because keys arrive with real delay between them.
let chain = Promise.resolve();

rl.on('line', (line) => {
  chain = chain.then(async () => {
    const sp = line.trim().indexOf(' ');
    const cmd = sp === -1 ? line.trim() : line.trim().slice(0, sp);
    const rest = sp === -1 ? '' : line.trim().slice(sp + 1);
    if (!cmd) return rl.prompt();
    const fn = COMMANDS[cmd];
    if (!fn) { console.log('unknown:', cmd, '- try: help'); return rl.prompt(); }
    try { await fn(rest); } catch (e) { console.log('ERROR:', e.message); }
    if (cmd === 'quit') { rl.close(); process.exit(0); }
    rl.prompt();
  });
});
// Piped stdin hits EOF (and fires 'close') the instant the heredoc ends,
// which races ahead of the still-pending `chain` of queued line handlers
// above. Await the chain here or `quit`/exit fires before `launch` etc.
// ever actually ran.
rl.on('close', async () => { await chain; await COMMANDS.quit(); process.exit(0); });

console.log('freqpred dashboard driver - "help" for commands, "launch" to start');
rl.prompt();
