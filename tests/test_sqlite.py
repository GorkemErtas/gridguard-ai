import sqlite3

from gridguard.cli import export_sqlite
from gridguard.synthetic import save_synthetic


def test_export_sqlite(tmp_path):
    source = save_synthetic(tmp_path / "demo.csv", customers=100, days=60, seed=2)
    destination = tmp_path / "demo.sqlite"
    assert export_sqlite(str(source), str(destination), limit_customers=25) == 25
    with sqlite3.connect(destination) as connection:
        count = connection.execute("SELECT COUNT(*) FROM consumption").fetchone()[0]
        ids = connection.execute(
            "SELECT COUNT(DISTINCT customer_id) FROM consumption"
        ).fetchone()[0]
    assert 0 < count <= 25 * 60
    assert ids == 25
