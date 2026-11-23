"""SQLite 版本化存储。

设计要点：
* 每个调查单（survey）有若干不可变版本；整单更正即追加新版本。
* 估计成功的版本成为现行版本（is_current=1），换版在单个事务内完成，
  中途失败不会留下半套状态。
* 估计失败（无可行配对、数据非法）也写入 status=failed 的版本并保留
  原始提交内容，且不改动 is_current 指针——失败后仍可查到原版结论。
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS surveys (
    survey_id   TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS survey_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id   TEXT NOT NULL REFERENCES surveys(survey_id),
    version     INTEGER NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    result      TEXT,
    error       TEXT,
    is_current  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (survey_id, version)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CurrentVersionMismatch(Exception):
    """发布时现行版号与请求期望不一致（乐观并发冲突），本次不留新版本。"""

    def __init__(self, expected: int | None, actual: int | None):
        self.expected = expected
        self.actual = actual
        if actual is None:
            hint = "该调查单尚无现行版本，expected_current_version 应填 null"
        elif expected is None:
            hint = f"该调查单现行版本已是 v{actual}，expected_current_version 不能填 null"
        else:
            hint = f"现行版本已变为 v{actual}，与请求期望的 v{expected} 不一致"
        super().__init__(f"现行版号冲突：{hint}；本次未写入新版本")


class SaveConflictError(Exception):
    """并发写入导致保存失败（如版本号唯一约束）；本次不留新版本。"""


class SurveyStore:
    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # WAL 允许读请求与写版本并发，busy_timeout 避免短暂锁冲突报错
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def save_version(
        self,
        survey_id: str,
        payload: dict,
        result: dict | None,
        error: str | None,
    ) -> dict:
        """追加一个版本；成功则切换现行指针，失败则保持原指针。事务内完成。"""
        status = "ok" if error is None else "failed"
        created_at = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO surveys(survey_id, created_at) VALUES (?, ?)",
                (survey_id, created_at),
            )
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM survey_versions WHERE survey_id=?",
                (survey_id,),
            ).fetchone()
            next_version = row["v"] + 1
            conn.execute(
                "INSERT INTO survey_versions"
                "(survey_id, version, status, created_at, payload, result, error, is_current) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    survey_id,
                    next_version,
                    status,
                    created_at,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    error,
                ),
            )
            if status == "ok":
                # 整单换版：旧版本全部归档，仅新版本为现行
                conn.execute(
                    "UPDATE survey_versions SET is_current=0 WHERE survey_id=?", (survey_id,)
                )
                conn.execute(
                    "UPDATE survey_versions SET is_current=1 "
                    "WHERE survey_id=? AND version=?",
                    (survey_id, next_version),
                )
        return self.get_version(survey_id, next_version)  # type: ignore[return-value]

    def save_version_if_current(
        self,
        survey_id: str,
        expected_current_version: int | None,
        payload: dict,
        result: dict,
    ) -> dict:
        """仅当现行版号与期望一致时，追加 ok 版本并切换现行指针（单事务）。

        补录复核的发布路径：计算期间同单可继续更正，本方法在提交前用
        BEGIN IMMEDIATE 取写锁并核对现行版号；版号冲突或保存失败时整体
        回滚，本次不留新版本。
        """
        created_at = _now()
        conn = self._connect()
        conn.isolation_level = None  # 手动管理事务
        next_version = -1
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT version FROM survey_versions WHERE survey_id=? AND is_current=1",
                (survey_id,),
            ).fetchone()
            actual = row["version"] if row is not None else None
            if actual != expected_current_version:
                raise CurrentVersionMismatch(expected_current_version, actual)
            conn.execute(
                "INSERT OR IGNORE INTO surveys(survey_id, created_at) VALUES (?, ?)",
                (survey_id, created_at),
            )
            next_version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM survey_versions WHERE survey_id=?",
                (survey_id,),
            ).fetchone()["v"]
            conn.execute(
                "INSERT INTO survey_versions"
                "(survey_id, version, status, created_at, payload, result, error, is_current) "
                "VALUES (?, ?, 'ok', ?, ?, ?, NULL, 0)",
                (
                    survey_id,
                    next_version,
                    created_at,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE survey_versions SET is_current=0 WHERE survey_id=?", (survey_id,)
            )
            conn.execute(
                "UPDATE survey_versions SET is_current=1 WHERE survey_id=? AND version=?",
                (survey_id, next_version),
            )
            conn.execute("COMMIT")
        except CurrentVersionMismatch:
            conn.execute("ROLLBACK")
            raise
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise SaveConflictError(f"保存失败（并发写入冲突）：{exc}；本次未写入新版本") from exc
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
        return self.get_version(survey_id, next_version)  # type: ignore[return-value]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "survey_id": row["survey_id"],
            "version": row["version"],
            "status": row["status"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload"]),
            "result": json.loads(row["result"]) if row["result"] is not None else None,
            "error": row["error"],
            "is_current": bool(row["is_current"]),
        }

    def get_version(self, survey_id: str, version: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM survey_versions WHERE survey_id=? AND version=?",
                (survey_id, version),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_current(self, survey_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM survey_versions WHERE survey_id=? AND is_current=1",
                (survey_id,),
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_versions(self, survey_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM survey_versions WHERE survey_id=? ORDER BY version",
                (survey_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]
