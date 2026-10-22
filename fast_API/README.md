# Student Directory API

A structured FastAPI example backed by `data.json`, which contains 1,000 synthetic student records.

## Run locally

```powershell
.\fast\Scripts\Activate.ps1
python main.py
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation.

## Routes

- `GET /` — API overview
- `GET /health` — health check
- `GET /api/v1/students?page=1&page_size=25` — paginated students
- `GET /api/v1/students/dataframe?page=1&page_size=25` — students serialized from a pandas DataFrame (`columns` and `data`)
- `GET /api/v1/students?course=Computer%20Science` — filter by course
- `GET /api/v1/students?search=ali` — search student names and emails
- `GET /api/v1/students/stats` — dataset counts and average GPA
- `GET /api/v1/students/{student_id}` — one student by ID
