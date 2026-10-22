from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.student import Student, StudentDataFrame, StudentPage, StudentStats

DATA_FILE = Path(__file__).resolve().parents[2] / "data.json"


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    """Load the JSON dataset once and keep it in a pandas DataFrame."""
    return pd.read_json(DATA_FILE)


def _filter_students(course: str | None = None, search: str | None = None) -> pd.DataFrame:
    frame = _load_dataframe()
    if course:
        frame = frame[frame["course"].str.casefold() == course.casefold()]
    if search:
        term = search.casefold()
        matches = frame["name"].str.casefold().str.contains(term, regex=False)
        matches |= frame["email"].str.casefold().str.contains(term, regex=False)
        frame = frame[matches]
    return frame


def list_students(
    page: int = 1,
    page_size: int = 25,
    course: str | None = None,
    search: str | None = None,
) -> StudentPage:
    frame = _filter_students(course=course, search=search)
    total = len(frame)
    start = (page - 1) * page_size
    records = frame.iloc[start : start + page_size].to_dict(orient="records")
    return StudentPage(
        items=[Student.model_validate(row) for row in records],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


def student_dataframe(
    page: int = 1,
    page_size: int = 25,
    course: str | None = None,
    search: str | None = None,
) -> StudentDataFrame:
    """Return a page from the pandas DataFrame in split orientation."""
    frame = _filter_students(course=course, search=search)
    total = len(frame)
    start = (page - 1) * page_size
    split_data: dict[str, Any] = frame.iloc[start : start + page_size].to_dict(orient="split")
    return StudentDataFrame(
        columns=split_data["columns"],
        data=split_data["data"],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_student(student_id: int) -> Student | None:
    frame = _load_dataframe()
    matches = frame.loc[frame["id"] == student_id]
    if matches.empty:
        return None
    return Student.model_validate(matches.iloc[0].to_dict())


def student_stats() -> StudentStats:
    frame = _load_dataframe()
    courses = {str(name): int(count) for name, count in frame["course"].value_counts().items()}
    statuses = {str(name): int(count) for name, count in frame["status"].value_counts().items()}
    average_gpa = float(frame["gpa"].mean()) if not frame.empty else 0.0
    return StudentStats(
        total_students=len(frame),
        average_gpa=round(average_gpa, 2),
        courses=courses,
        statuses=statuses,
    )
