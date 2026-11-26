"""Local run ledger. Retrieve prior experiments by task and goal similarity."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS runs "
                       "(id TEXT PRIMARY KEY, task TEXT NOT NULL, goal TEXT NOT NULL, report TEXT NOT NULL)")

    def remember(self, report):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                       (report["id"], report["task"], report["goal"], json.dumps(report, allow_nan=False)))

    def recall(self, goal, task):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT id, goal, report FROM runs WHERE task=? "
                              "ORDER BY rowid DESC LIMIT 200", (task,)).fetchall()
        if not rows:
            return []
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(
            [goal] + [row[1] for row in rows])
        scores = cosine_similarity(vectors[0], vectors[1:])[0]
        ranked = sorted(range(len(rows)), key=lambda i: -scores[i])
        return [{"id": rows[i][0], "goal": rows[i][1], "winner": json.loads(rows[i][2])["winner"]}
                for i in ranked[:3] if scores[i] >= 0.15]
