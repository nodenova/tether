# Testing Setup Guide

How to set up any project for e2e testing with Tether's /test command and Playwright browser tools.

## Overview

Tether's /test command drives a 9-phase agent workflow that discovers your project, starts the dev server, runs tests, interacts with the live UI through a real browser, finds bugs, fixes them, and reports results. The browser automation is powered by Playwright MCP -- Claude Code controls a headless Chromium instance to navigate pages, click elements, fill forms, and verify state.

This guide covers three tiers of setup, from zero-config to full AI-assisted test authoring.

---

## Tier 1: Basic Setup (Zero Changes to Your Repo)

The /test command works out of the box with any project. The agent uses Playwright MCP browser tools directly -- no Playwright files need to exist in your repo.

### Machine Prerequisites

| Requirement | Why | Install |
|---|---|---|
| Node.js 18+ | Playwright MCP runs via npx | nodejs.org |
| Chromium binary | Headless browser for page interaction | `npx playwright install chromium` (one-time) |

The Chromium install prints a warning about missing project dependencies. Safe to ignore -- the binary installs successfully.

### Tether Configuration

Your project directory must be in the approved directories list:

```bash
TETHER_APPROVED_DIRECTORIES=/path/to/your/project,/path/to/other/project
```

Switch between projects with /dir:

```
/dir                    # List available directories
/dir myproject          # Switch to myproject
```

### MCP Server

Tether's .mcp.json configures the Playwright MCP server automatically:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@0.0.41", "--headless"]
    }
  }
}
```

This spawns a headless Chromium instance managed by Claude Code's SDK. Your project needs no MCP configuration -- the browser navigates to URLs served by your dev server.

For visual debugging, switch to headed mode:

```json
"args": ["@playwright/mcp@0.0.41", "--headed"]
```

### Running /test

Basic invocation -- the agent discovers everything automatically:

```
/test
```

With hints to guide the agent:

```
/test --url http://localhost:5173
/test --server "npm run dev" --url http://localhost:3000
/test --framework next.js
/test --no-unit authentication flow
/test --no-e2e --no-backend
```

#### /test Command Reference

| Flag | Short | Description | Example |
|---|---|---|---|
| --url | -u | App URL for browser testing | `--url http://localhost:8080` |
| --server | -s | Dev server startup command | `--server "npm run dev"` |
| --dir | -d | Test directory hint | `--dir tests/e2e` |
| --framework | -f | Framework hint | `--framework react` |
| --no-e2e | | Skip browser testing phases | |
| --no-unit | | Skip unit/integration test phase | |
| --no-backend | | Skip backend API verification | |
| (free text) | | Focus area for the agent | `login page validation` |

Flags combine: `/test --url http://localhost:3000 --no-backend checkout flow`

### The 9-Phase Workflow

| Phase | Name | What happens |
|---|---|---|
| 1 | Discovery | Reads package.json, pyproject.toml, Cargo.toml, go.mod. Identifies frameworks, dev server commands, recent changes (git diff). |
| 2 | Server Startup | Starts the dev server (detected or via --server). Waits for readiness with curl/lsof. Fixes startup errors. |
| 3 | Smoke Test | Navigates to app URL. Verifies page loads. Checks console for JS errors. Takes accessibility snapshot. |
| 4 | Unit and Integration | Runs existing suites (pytest, jest, vitest, go test, cargo test). Analyzes and fixes failures. |
| 5 | Backend Verification | Hits API endpoints with curl. Checks responses, logs, error handling. |
| 6 | E2E Browser Testing | Interacts with live UI via browser tools. Tests forms, navigation, auth, error states. Writes test files if none exist. |
| 7 | Error Analysis | Compiles all errors. Categorizes as CRITICAL / HIGH / MEDIUM / LOW. |
| 8 | Healing | Fixes bugs from previous phases. Re-runs affected tests. |
| 9 | Report | Summary: pass/fail counts, errors, fixes applied, remaining issues, health assessment. |

Phases 2, 3, 6 skip with --no-e2e. Phase 4 skips with --no-unit. Phase 5 skips with --no-backend.

### What Gets Auto-Approved in Test Mode

When /test activates, the following tools are auto-approved (no human confirmation):

All 25 browser tools -- readonly and mutation.

Test-related bash commands:

```
npx playwright, npx jest, npx vitest, npx mocha, npx tsc
npm run, npm test, npm start, npm exec
yarn run, yarn test, yarn start
pnpm run, pnpm test, pnpm start
uv run pytest, uv run python, uv run ruff
pytest, python, node
go test, cargo test
curl, wget, lsof, kill
cat, ls, head, tail, wc, grep, find
```

File writes -- Write and Edit tools auto-approved for test file creation and source fixes.

### Browser Tools Reference

The agent has 25 browser tools via Playwright MCP:

Observation (7 tools, readonly):

| Tool | What it does |
|---|---|
| browser_snapshot | Page accessibility tree (text-based, compact) |
| browser_take_screenshot | PNG screenshot (expensive in tokens) |
| browser_console_messages | Browser console log messages |
| browser_network_requests | Network requests made by the page |
| browser_tab_list | Open browser tabs |
| browser_wait_for | Wait for text, element, or timeout |
| browser_generate_playwright_test | Generate test from recorded actions |

Interaction (18 tools, mutation):

| Tool | What it does |
|---|---|
| browser_navigate | Go to a URL |
| browser_click | Click an element |
| browser_type | Type text into an input |
| browser_hover | Hover over an element |
| browser_drag | Drag and drop between elements |
| browser_press_key | Press a keyboard key |
| browser_select_option | Select a dropdown option |
| browser_file_upload | Upload a file |
| browser_handle_dialog | Accept or dismiss a dialog |
| browser_navigate_back | Browser back button |
| browser_navigate_forward | Browser forward button |
| browser_tab_new | Open a new tab |
| browser_tab_select | Switch to a tab |
| browser_tab_close | Close a tab |
| browser_resize | Resize the viewport |
| browser_pdf_save | Save page as PDF |
| browser_close | Close the browser |
| browser_install | Install a browser |

The agent prefers browser_snapshot over browser_take_screenshot -- faster, cheaper, structured data. Screenshots reserved for visual layout issues.

---

## Tier 2: Persistent Playwright Tests in Your Repo

For saved, re-runnable e2e test files (.spec.ts) that live in your project and run independently of Tether.

### Setup Steps

1. Create a minimal package.json (or add to existing):

```json
{
  "private": true,
  "devDependencies": {
    "@playwright/test": "^1.50.0"
  }
}
```

2. Install dependencies and browser:

```bash
npm install
npx playwright install chromium
```

3. Create playwright.config.ts at project root:

```typescript
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",

  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,

  reporter: [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: process.env.BASE_URL || "http://localhost:3000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
```

Adjust baseURL to your dev server port. BASE_URL env var overrides at runtime.

4. Create e2e/ directory with a seed test:

```bash
mkdir e2e
```

```typescript
// e2e/smoke.spec.ts
import { test, expect } from "@playwright/test";

test.describe("smoke tests", () => {
  test("homepage loads", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/.+/);
  });

  test("no console errors on load", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") errors.push(msg.text());
    });

    await page.goto("/");
    await page.waitForLoadState("networkidle");
    expect(errors).toEqual([]);
  });
});
```

5. Add to .gitignore:

```
node_modules/
test-results/
playwright-report/
blob-report/
```

6. Run tests:

```bash
npm run dev              # start dev server first
npx playwright test      # run all e2e tests
npx playwright test --headed   # with visible browser
npx playwright test --ui       # interactive UI mode
npx playwright show-report     # view HTML report
```

### How It Connects to /test

When /test enters Phase 6 (E2E Browser Testing), the agent:

1. Detects playwright.config.ts during Phase 1 (Discovery)
2. Runs existing tests with `npx playwright test`
3. Analyzes failures and fixes broken tests
4. Writes new test files for untested flows in e2e/

Without Tier 2, the agent still tests via browser tools -- it just does not persist tests as .spec.ts files (unless created ad-hoc during Phase 6).

---

## Tier 3: AI Test Agents (Planner, Generator, Healer)

Playwright provides three AI agents for Claude Code sessions. Requires Tier 2 (Playwright Test framework in place).

### Setup

Run in your project directory:

```bash
npx playwright init-agents --loop=claude
```

Creates .agents/ directory with:

| Agent | Purpose |
|---|---|
| Planner | Analyzes your app, creates a structured test plan |
| Generator | Generates Playwright test code from the plan |
| Healer | Finds and fixes broken tests by inspecting live UI |

### Using the Agents

Planner -- ask the agent to analyze and plan:
> "Analyze the app at localhost:3000 and create a test plan covering all user flows"

Generator -- ask the agent to write tests:
> "Generate Playwright tests for the authentication flow based on the test plan"

Healer -- use the /healer slash command:
1. Runs `npx playwright test` to identify failures
2. Analyzes Playwright trace files for each failure
3. Uses browser_snapshot to inspect current UI state
4. Compares expected vs actual state
5. Fixes test code (selectors, assertions, waits)
6. Re-runs to confirm the fix

The /healer command works independently of /test -- use it anytime Playwright tests are failing.

### Agent Output

Generated test files can be:
- Run with `npx playwright test`
- Committed to your repo
- Included in CI/CD pipelines
- Maintained with the Healer when UI changes

---

## Policy Behavior

Outside test mode, browser tools are gated by safety policies:

| Policy | Readonly tools | Mutation tools |
|---|---|---|
| default | Auto-allow | Require approval |
| strict | Require approval | Require approval |
| permissive | Auto-allow | Auto-allow |

In test mode (/test), all 25 browser tools are auto-approved regardless of policy. Test mode also auto-approves file writes for test creation and source fixes.

Test mode scopes to the current chat session. Resets on /dir switch or new session.

---

## Token Cost Considerations

- browser_snapshot: accessibility tree, text-based, compact. Use for verifying page state.
- browser_take_screenshot: PNG image, expensive in tokens and context window. Reserve for visual layout debugging.

The agent defaults to browser_snapshot. For large test sessions, focus with `/test authentication flow` rather than running everything.

---

## Troubleshooting

**Dev server not running** --
Provide exact command and URL:
```
/test --server "npm run dev" --url http://localhost:3000
```

**Port already in use** --
The agent checks with lsof. Force a specific port:
```
/test --url http://localhost:8080
```

**Headless limitations** --
OS file pickers, system notifications, and browser extensions need headed mode. Switch .mcp.json to --headed temporarily.

**Context window exhaustion** --
Focus: `/test login page`. Skip phases: `--no-unit --no-backend`. The agent uses browser_snapshot over screenshots to conserve tokens.

**Tests pass locally but fail via Tether** --
Playwright MCP runs a separate headless Chromium. Viewport, fonts, and extensions may differ. Use browser_resize to match expected dimensions.

**Chromium not installed** --
```bash
npx playwright install chromium
```

---

## Quick Reference

```bash
# Machine setup (one-time)
npx playwright install chromium

# Basic testing (no repo changes)
/test
/test --url http://localhost:5173 --server "npm run dev"

# Saved Playwright tests (Tier 2)
npx playwright test
npx playwright test --headed
npx playwright test --ui
npx playwright show-report

# AI test agents (Tier 3)
npx playwright init-agents --loop=claude

# Fix broken tests
/healer
```

See also: [Browser Testing](browser-testing.md) for Playwright MCP configuration details and known gotchas.
