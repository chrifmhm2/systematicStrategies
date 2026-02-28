# Phase 7 — Polish & Deploy

> **Goal**: Containerize the application, set up CI/CD, deploy to the internet, and write the README. By the end you have a live public URL for both frontend and backend.

**Prerequisites**: Phases 1–6 complete and all tests passing.

---

## Docker

- [ ] **[P7-01]** Create `backend/Dockerfile`:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY pyproject.toml .
  RUN pip install --no-cache-dir .
  COPY . .
  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- [ ] **[P7-02]** Create `frontend/Dockerfile` (multi-stage build):
  ```dockerfile
  FROM node:20-alpine AS build
  WORKDIR /app
  COPY package*.json .
  RUN npm ci
  COPY . .
  RUN npm run build

  FROM nginx:alpine
  COPY --from=build /app/dist /usr/share/nginx/html
  ```
- [ ] **[P7-03]** Create `frontend/nginx.conf` that rewrites all routes to `index.html` (required for React Router client-side routing)
- [ ] **[P7-04]** Create `docker-compose.yml` at the repo root:
  ```yaml
  services:
    backend:
      build: ./backend
      ports: ["8000:8000"]
      environment:
        - PYTHONDONTWRITEBYTECODE=1
    frontend:
      build: ./frontend
      ports: ["3000:80"]
      depends_on: [backend]
  ```
- [ ] **[P7-05]** Verify `docker-compose up --build` starts both containers and the frontend is reachable at `http://localhost:3000`

---

## CORS & Environment Configuration

- [ ] **[P7-06]** Update `backend/config.py` to read `ALLOWED_ORIGINS` from an environment variable (comma-separated); default to `["*"]` for development
- [ ] **[P7-07]** Update `CORSMiddleware` in `backend/main.py` to use `settings.ALLOWED_ORIGINS`
- [ ] **[P7-08]** Create `frontend/.env.example`:
  ```
  VITE_API_URL=http://localhost:8000
  ```
- [ ] **[P7-09]** Update `src/api/client.ts` to use `import.meta.env.VITE_API_URL` as the Axios `baseURL` (fallback to `""` for proxied dev)

---

## CI/CD (`.github/workflows/`)

- [ ] **[P7-10]** Create `.github/workflows/backend-ci.yml`:
  - Trigger: push to `main`, pull requests
  - Steps: checkout, setup Python 3.12, `pip install -e ".[dev]"`, `ruff check .`, `pytest --cov=core --cov-fail-under=80`
- [ ] **[P7-11]** Create `.github/workflows/frontend-ci.yml`:
  - Trigger: push to `main`, pull requests
  - Steps: checkout, setup Node 20, `npm ci`, `npm run lint`, `npm run build`
- [ ] **[P7-12]** Verify both workflows pass on the `main` branch (check the GitHub Actions tab)

---

## Backend Deployment (Railway or Render)

- [ ] **[P7-13]** Create a new service on Railway (or Render) pointing to the `backend/` directory
- [ ] **[P7-14]** Set environment variables: `ALLOWED_ORIGINS=<your-vercel-domain>`
- [ ] **[P7-15]** Confirm the deployed backend is reachable at its public URL (e.g. `https://quantforge-api.railway.app/docs`)

---

## Frontend Deployment (Vercel)

- [ ] **[P7-16]** Connect the GitHub repo to Vercel, set root directory to `frontend/`
- [ ] **[P7-17]** Set environment variable `VITE_API_URL=<your-railway-backend-url>` in Vercel project settings
- [ ] **[P7-18]** Deploy and confirm the live URL renders the dashboard and can call the API

---

## README (`README.md`)

- [ ] **[P7-19]** Write `README.md` at the repo root with:
  - **[P7-19a]** Project title "QuantForge" + one-line description
  - **[P7-19b]** Live demo link (Vercel URL)
  - **[P7-19c]** Screenshot of the Strategy Comparison page
  - **[P7-19d]** Features section (4 bullet points)
  - **[P7-19e]** Architecture diagram (embed `01_Projet_DotNET/res/conception/conception.svg` or create a new one)
  - **[P7-19f]** Tech stack badges (shields.io: Python, FastAPI, React, TypeScript, Tailwind, Docker)
  - **[P7-19g]** Quick start section:
    ```bash
    docker-compose up --build
    # Frontend: http://localhost:3000
    # Backend API docs: http://localhost:8000/docs
    ```
  - **[P7-19h]** Strategies table: 8 rows with Strategy | Family | Key parameter columns
  - **[P7-19i]** API documentation link pointing to `/docs`
  - **[P7-19j]** Testing section: `pytest` and `npm test` commands
  - **[P7-19k]** Acknowledgments: Ensimag course, Professor Mnacho Echenim

---

## Final QA Checklist

- [ ] **[P7-20]** Run the full backend test suite: `pytest tests/ --cov=core --cov-fail-under=80` — all pass
- [ ] **[P7-21]** Run `ruff check backend/` — no linting errors
- [ ] **[P7-22]** Run `npm run lint` in `frontend/` — no ESLint errors
- [ ] **[P7-23]** Run `npm run build` in `frontend/` — production build succeeds with no warnings
- [ ] **[P7-24]** Manual smoke test on the live URL:
  - Home page loads
  - Strategy Explorer shows 6+ strategies
  - Running a backtest on 3 assets returns an equity curve
  - Hedging Simulator runs with simulated data
  - Risk Analytics page shows VaR histogram

---

## Definition of Done

- `docker-compose up --build` works and the full app runs locally
- CI pipelines are green on `main`
- Both frontend and backend are deployed and accessible via public URLs
- `README.md` contains the live demo link and a screenshot
