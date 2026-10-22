from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Student(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    age: int = Field(ge=16, le=100)
    course: str
    department: str
    email: str
    gpa: float = Field(ge=0, le=4)
    enrollment_year: int
    status: str


class StudentPage(BaseModel):
    items: list[Student]
    total: int
    page: int
    page_size: int
    pages: int


class StudentDataFrame(BaseModel):
    """A paginated DataFrame serialized using pandas split orientation."""

    columns: list[str]
    data: list[list[Any]]
    total: int
    page: int
    page_size: int


class StudentStats(BaseModel):
    total_students: int
    average_gpa: float
    courses: dict[str, int]
    statuses: dict[str, int]
