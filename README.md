# CrimeHotspotProject

Development README

Conda environment (recommended)

1. Create and activate the conda env:

```powershell
C:\Users\Dell\Miniconda3\Scripts\conda.exe create -n crimehotspot python=3.11 -y
C:\Users\Dell\Miniconda3\Scripts\conda.exe activate crimehotspot
```

2. Install dependencies (conda for numeric libs, pip for web packages):

```powershell
C:\Users\Dell\Miniconda3\Scripts\conda.exe install -n crimehotspot -c conda-forge -y numpy scipy scikit-learn
C:\Users\Dell\Miniconda3\Scripts\conda.exe run -n crimehotspot python -m pip install fastapi uvicorn pydantic sqlalchemy
```

3. Create and populate the database (SQLite by default):

```powershell
C:\Users\Dell\Miniconda3\Scripts\conda.exe run -n crimehotspot python database_models.py
```

# CrimeHotspotProject

This repository contains a FastAPI backend and a Streamlit frontend for crime hotspot mapping, risk prediction, and patrol recommendations.

This README provides a concise step-by-step guide to set up, run, and smoke-test the application on Windows using PowerShell and the Conda environment (recommended name: `crimehotspot_new`).

## Prerequisites
- Windows with PowerShell
- Miniconda/Anaconda (or another Python environment manager)

## 1) Create / activate the Conda environment
If you don't already have a suitable environment, create one and activate it. Example:

```powershell
conda create -n crimehotspot_new python=3.9 -y
conda activate crimehotspot_new
```

If `conda activate` doesn't work directly in PowerShell, use the full conda.bat path:

```powershell
& 'C:\Users\<your_user>\Miniconda3\condabin\conda.bat' activate crimehotspot_new
```

## 2) Install dependencies
Install required Python packages into the active environment. If a `requirements.txt` is present, use it:

```powershell
python -m pip install -r requirements.txt
```

Or install common runtime packages manually:

```powershell
python -m pip install fastapi uvicorn[standard] streamlit sqlalchemy alembic scikit-learn joblib pandas numpy requests plotly pydantic pydantic-settings
```

## 3) Database initialization
On first run, the FastAPI app will automatically create the SQLite file at `data/crimehotspot.db`.

Optional: generate and apply an Alembic migration:

```powershell
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

## 4) Start the backend (FastAPI)
From the project root in PowerShell run:

```powershell
# If the environment is active
python -m uvicorn app.main:app --port 8000

# Or use an explicit python path (replace USERNAME as needed)
& 'C:\Users\<your_user>\Miniconda3\envs\crimehotspot_new\python.exe' -m uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000/docs` to view the OpenAPI docs.

## 5) Start the frontend (Streamlit)
In a separate PowerShell session (with the same conda env active):

```powershell
python -m streamlit run frontend/frontend.py
```

Streamlit will open a local URL in your browser.

## 6) Smoke tests (quick)
Run a small Python snippet to verify key endpoints:

```powershell
python - <<'PY'
import requests, json
base='http://127.0.0.1:8000/api'
endpoints=['status','crime-stats','realtime-crimes','patrol-recommendations','hotspots','risk-predictions','crimes']
for e in endpoints:
	try:
		r=requests.get(f"{base}/{e}", timeout=5)
		print(e, r.status_code)
		try:
			print(json.dumps(r.json(), indent=2)[:500])
		except Exception as ex:
			print('no json:', ex)
	except Exception as ex:
		print(e, 'ERROR', ex)
PY
```

## 7) Troubleshooting
- If you see Pydantic errors about BaseSettings, install `pydantic-settings`:

```powershell
python -m pip install pydantic-settings
```

- Make sure you run server commands from the project root so the `app` package is importable.

## 8) Next steps
- Add authentication (JWT), structured logging, tests, and CI.
- Generate Alembic migration(s) and commit them.

If you'd like, I can now (a) start the backend in a background terminal and re-run the smoke tests, (b) start Streamlit for a quick UI check, or (c) continue with further cleanup and tests. Pick one and I'll proceed.
