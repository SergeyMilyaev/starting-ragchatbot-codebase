# Frontend Code Quality Tooling

## What was added

### New files
- `frontend/package.json` — npm package manifest with `format`, `format:check`, `lint`, and `check` scripts
- `frontend/.prettierrc` — Prettier config: single quotes, 2-space indent, trailing commas, 100-char print width
- `frontend/.prettierignore` — excludes `node_modules/`
- `frontend/eslint.config.js` — ESLint flat config for `script.js` with browser globals and sensible rules (`no-unused-vars`, `eqeqeq`, `no-var`, `prefer-const`)
- `frontend/check.sh` — convenience shell script to run `npm run check` from any working directory

### Modified files
`script.js`, `style.css`, and `index.html` were reformatted by Prettier:
- Indentation normalized from 4 spaces to 2 spaces throughout
- Trailing whitespace and double blank lines removed
- Consistent quote style (single quotes in JS)
- Arrow function parameter parens added where Prettier requires them (e.g. `forEach(button =>` → `forEach((button) =>`)

## How to use

```bash
# Format all frontend files
cd frontend && npm run format

# Check formatting without modifying files
cd frontend && npm run format:check

# Lint JavaScript
cd frontend && npm run lint

# Run all checks at once
cd frontend && npm run check
# or
./frontend/check.sh
```

## Tooling details

| Tool | Version | Purpose |
|------|---------|---------|
| Prettier | ^3.0.0 | Formats JS, CSS, HTML |
| ESLint | ^9.0.0 | Lints JavaScript for logic errors and style rules |

All checks pass on the current codebase.
