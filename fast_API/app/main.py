from fastapi import FastAPI

from app.api.routes import students

app = FastAPI(
    title="Student Directory API",
    description="A paginated API for browsing a large synthetic student dataset.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(students.router, prefix="/api/v1")


@app.get("/", tags=["General"])
def read_root() -> dict[str, str]:
    return {
        "name": "Student Directory API",
        "version": app.version,
        "docs": "/docs",
        "students": "/api/v1/students",
    }


@app.get("/health", tags=["General"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
