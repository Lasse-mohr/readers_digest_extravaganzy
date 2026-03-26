from datetime import date, datetime
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, JSON, Column
from sqlalchemy import JSON as SAJSON

DATABASE_URL = "sqlite:///./digest.db"
engine = create_engine(DATABASE_URL, echo=False)


class Paper(SQLModel, table=True):
    id: str = Field(primary_key=True)  # DOI if available, else arXiv ID
    title: str
    abstract: Optional[str] = None
    authors: list = Field(default_factory=list, sa_column=Column(SAJSON))
    journal: Optional[str] = None
    published_date: date
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None
    source: str  # "openalex" | "arxiv"
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class DigestRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    window_start: date
    window_end: date
    papers_fetched: int = 0
    papers_included: int = 0
    markdown_path: Optional[str] = None
    status: str = "running"  # "running" | "complete" | "error"


class ScoredPaper(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: str = Field(foreign_key="paper.id")
    digest_run_id: int = Field(foreign_key="digestrun.id")
    similarity_score: float = 0.0
    final_score: float = 0.0
    priority_author_match: bool = False
    bluesky_mentions: list = Field(default_factory=list, sa_column=Column(SAJSON))
    summary: Optional[str] = None
    included_in_digest: bool = False


class CoauthorCache(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    anchor_openalex_id: str = Field(index=True)
    coauthor_openalex_id: str
    coauthor_name: str
    collaboration_count: int = 0
    cached_at: datetime = Field(default_factory=datetime.utcnow)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
