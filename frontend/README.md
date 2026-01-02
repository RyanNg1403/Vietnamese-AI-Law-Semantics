# Query Visualization Frontend

## Overview

Interactive visualization system for legal text queries with full pipeline tracing:
- **WSD** (Word Sense Disambiguation) - MFS or BERT
- **FOL Translation** (First-Order Logic)
- **Prolog Inference** with step-by-step details

## Files

- `query.html` - Main HTML page
- `query-style.css` - Dark mode styling
- `query-script.js` - Frontend logic

## Usage

1. Start the FastAPI backend:
```bash
cd backend
python query_api.py
```

2. Open `query.html` in a browser

3. Select a query from the dropdown

4. Choose WSD method (MFS or BERT)

5. Click "Run Query" to see the full pipeline visualization

## Features

- ✅ Dark mode UI
- ✅ Pipeline animation
- ✅ Step-by-step details (accordion)
- ✅ WSD results table
- ✅ Prolog inference tracing
- ✅ Final result display

## API Endpoints

- `GET /queries` - Get all predefined queries
- `POST /query` - Process query with tracing

