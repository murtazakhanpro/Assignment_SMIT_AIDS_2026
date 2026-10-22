from fastapi import APIRouter, HTTPException, Query

from app.schemas.student import Student, StudentDataFrame, StudentPage, StudentStats
from app.services.student_service import (
    get_student,
    list_students,
    student_dataframe,
    student_stats,
)

router = APIRouter(prefix="/students", tags=["Students"])


@router.get("", response_model=StudentPage, summary="List students")
def read_students(
    page: int = Query(1, ge=1, description="Page number, starting at 1"),
    page_size: int = Query(25, ge=1, le=100, description="Results per page"),
    course: str | None = Query(None, description="Filter by course name"),
    search: str | None = Query(None, min_length=1, description="Search name or email"),
) -> StudentPage:
    return list_students(page=page, page_size=page_size, course=course, search=search)


@router.get("/dataframe", response_model=StudentDataFrame, summary="Get students as a DataFrame")
def read_students_dataframe(
    page: int = Query(1, ge=1, description="Page number, starting at 1"),
    page_size: int = Query(25, ge=1, le=100, description="Rows per page"),
    course: str | None = Query(None, description="Filter by course name"),
    search: str | None = Query(None, min_length=1, description="Search name or email"),
) -> StudentDataFrame:
    return student_dataframe(page=page, page_size=page_size, course=course, search=search)


@router.get("/stats", response_model=StudentStats, summary="Get dataset statistics")
def read_student_stats() -> StudentStats:
    return student_stats()


@router.get("/{student_id}", response_model=Student, summary="Get a student")
def read_student(student_id: int) -> Student:
    student = get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student
