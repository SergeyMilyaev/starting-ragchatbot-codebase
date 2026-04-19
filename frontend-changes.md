# Frontend Changes

## Backend Testing Framework

No frontend changes were made in this task.

This task enhanced the **backend testing framework** only:

- `backend/tests/conftest.py` — added `mock_rag_system` and `api_client` fixtures; sys.path now includes both `backend/` and `backend/tests/`
- `backend/tests/_helpers.py` — new helper module with `build_test_app()` factory that creates a minimal FastAPI app (same endpoints as `app.py`, no static file mount) wired to a provided RAG instance
- `backend/tests/test_api_endpoints.py` — 17 new tests covering `POST /api/query`, `GET /api/courses`, and `DELETE /api/session/{session_id}` (happy path, error cases, and validation)
- `pyproject.toml` — added `httpx>=0.27.0` dev dependency and `[tool.pytest.ini_options]` with `testpaths`, `pythonpath`, and `-v` flag

## Dark/Light Theme Toggle

### Files Modified
- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

### What Was Added

#### `index.html`
- Added a `<button id="themeToggle">` fixed to the top-right corner, containing two SVG icons: a sun (shown in light mode) and a moon (shown in dark mode).
- Bumped cache-busting version on `style.css` (v11) and `script.js` (v12).

#### `style.css`
- Added `[data-theme="light"]` CSS variable overrides on the `:root` selector's variables, providing a light colour palette (light backgrounds, dark text, adjusted borders and surfaces).
- Added `.theme-toggle` styles: fixed positioning top-right, circular button, border and background using theme variables, hover/focus ring using `--focus-ring`.
- Added icon visibility rules: `.icon-moon` visible by default (dark mode); `.icon-sun` visible only under `[data-theme="light"]`.
- Added `transition: background-color 0.3s ease, color 0.3s ease` to `body` for smooth theme switching.
- Added `transition` to `.sidebar` for background and border colour changes.
- Changed hardcoded link colours in `.sources-content a` to use `var(--primary-color)` and `var(--primary-hover)` so they adapt to the active theme.

#### `script.js`
- Added `initTheme()`: reads `localStorage.getItem('theme')` on load and applies `data-theme="light"` to `<html>` if saved as light.
- Added `toggleTheme()`: toggles `data-theme` attribute on `document.documentElement` and persists the choice to `localStorage`.
- Wired `toggleTheme` to the `#themeToggle` button's `click` event inside `setupEventListeners()`.
- Called `initTheme()` at the top of the `DOMContentLoaded` handler so the correct theme is applied before the page renders.

### Design Decisions
- Theme attribute is placed on `<html>` (`document.documentElement`) so CSS variable overrides apply globally.
- Default (no attribute) is dark mode to preserve the existing design.
- `localStorage` persists the user's preference across page reloads.
