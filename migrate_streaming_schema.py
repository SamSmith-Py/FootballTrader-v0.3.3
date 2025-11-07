import sqlite3
from core.settings import DB_PATH, TABLE_CURRENT, TABLE_STREAM

def table_has_column(conn, table, col):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(r[1].lower() == col.lower() for r in cur.fetchall())

def ensure_column(conn, table, col, sql_type):
    if not table_has_column(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type}")
        print(f"Added column {col} to {table}")

def ensure_stream_table(conn):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_STREAM} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            league TEXT,
            event_name TEXT,
            event_id TEXT NOT NULL,
            last_h_price REAL,
            h_price REAL,
            last_d_price REAL,
            d_price REAL,
            last_a_price REAL,
            a_price REAL,
            h_score INTEGER,
            a_score INTEGER,
            inplay_time INTEGER,    -- minutes if available
            h_red_cards INTEGER,
            a_red_cards INTEGER,
            timestamp TEXT NOT NULL  -- ISO string
        )
    """)
    # Indexes to speed up charts & filters
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_stream_event_time ON {TABLE_STREAM}(event_id, timestamp)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_stream_event_price ON {TABLE_STREAM}(event_id, h_price, d_price, a_price)")
    print(f"Ensured table {TABLE_STREAM} and indexes")

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Add rolling “last_*” price columns to current_matches (nullable)
        ensure_column(conn, TABLE_CURRENT, "last_h_price", "REAL")
        ensure_column(conn, TABLE_CURRENT, "last_d_price", "REAL")
        ensure_column(conn, TABLE_CURRENT, "last_a_price", "REAL")

        # Create stream history table + indexes
        ensure_stream_table(conn)

        conn.commit()
        print("Migration OK.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
