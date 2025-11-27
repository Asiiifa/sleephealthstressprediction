# Copilot Instructions for sleephealthstressprediction

## Project Overview
- This is a Flask web application for sleep, health, and stress prediction or tracking.
- The main entry point is `app.py`, which defines all routes and serves the frontend via `static/index.html`.
- User interaction is routed through four endpoints: `/`, `/sleep`, `/stress`, and `/history`.
- User history is stored in `static/history.csv` and displayed on the `/history` page.

## Key Files & Structure
- `app.py`: Flask app, all routes, and data loading logic.
- `static/index.html`: Main HTML template (currently empty, but all pages render this file with a `page` context variable).
- `static/history.csv`: Stores user history as a CSV, loaded and rendered as an HTML table on the `/history` route.
- `static/style.css`: CSS for the frontend (currently empty).
- `requirements.txt`: Should list Python dependencies (Flask, pandas, etc.). Currently empty—add dependencies as needed.

## Patterns & Conventions
- All HTML rendering uses `render_template` with `index.html` and a `page` variable to control content.
- Data for the `/history` page is loaded from `static/history.csv` using pandas, and rendered as an HTML table.
- No database is used; all persistent data is stored as CSV in the `static` directory.
- No custom Flask blueprints or API endpoints—everything is handled in a single file.

## Developer Workflows
- **Run the app:**
  ```pwsh
  python app.py
  ```
  The app runs in debug mode by default.
- **Add dependencies:**
  Add required packages to `requirements.txt` and install with:
  ```pwsh
  pip install -r requirements.txt
  ```
- **Update frontend:**
  Edit `static/index.html` and `static/style.css` as needed. All pages use the same HTML template.
- **View history:**
  Add or update `static/history.csv` to change the data shown on the `/history` page.

## Integration Points
- Uses Flask for routing and pandas for CSV handling. No other integrations or external APIs.
- All static files are under the `static/` directory.

## Examples
- To add a new page, define a new route in `app.py` and render `index.html` with a new `page` value.
- To change the data shown on `/history`, update `static/history.csv`.

## Recommendations
- Keep all persistent data in `static/history.csv` unless migrating to a database.
- Document any new routes or data files in this file for future contributors.
