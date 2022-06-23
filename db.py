import sqlite3
from typing import List, Optional, Tuple

DB_NAME: str = "files.db"
TABLE_NAME: str = "Files"
ENTITIES: List[str] = ["File_name", "Contents"]
DATA: List[Tuple[str, str]] = [("File_name", "TEXT"), ("Contents", "TEXT")]


def sql_execute(sql_string: str) -> str:
    """Execute provided SQL string on db, returns id of last row on cursor."""
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.cursor()
        cursor.execute(sql_string)
        connection.commit()
    return cursor.lastrowid


def create_table(table_name: str, entities: List[Tuple[str, str]]) -> None:
    """Create table if it does not exist already.

    :entities: List of tuples containing (entity_name, entity_format), e.g.
        ("Key", "INT PRIMARY_KEY")

    """
    entity_string = []
    for entity_name, *entity_format in entities:
        entity_string.append(f"[{entity_name}] {', '.join(entity_format)}")
    entity_string = ", ".join(entity_string)

    sql_string = f"CREATE TABLE IF NOT EXISTS {table_name} ({entity_string});"
    sql_execute(sql_string)


def drop_table(table_name: str) -> None:
    """Drop table if it exists."""
    sql_string = f"DROP TABLE IF EXISTS {table_name};"
    sql_execute(sql_string)


def add_entry(table_name: str, entities: List[str], values: List[str]) -> str:
    """Add entry to existing db table and return its id.

    :entities: List of entities for which to add values.
    :values: List of values to add.

    """
    entity_string = ", ".join([f"[{entity_name}]" for entity_name in entities])
    value_string = ", ".join([f"'{value}'" for value in values])

    sql_string = f"INSERT INTO {table_name} ({entity_string}) VALUES ({value_string});"
    entry_id = sql_execute(sql_string)
    return entry_id


def get_all_entries(table_name: str) -> List[tuple]:
    """Get entry from existing db table by id."""
    sql_string = f"SELECT ROWID, * FROM {table_name};"
    sql_execute(sql_string)
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.cursor()
        cursor.execute(sql_string)
    return cursor.fetchall()


def get_entry_by_id(table_name: str, rowid: int) -> tuple:
    """Get entry from existing db table by id."""
    sql_string = f"SELECT * FROM {table_name} WHERE ROWID = '{rowid}';"
    sql_execute(sql_string)
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.cursor()
        cursor.execute(sql_string)
    return cursor.fetchone()


def print_table(table_name: str, entities: Optional[List[str]] = None) -> None:
    """Print contents of db table, optionally filter by entities."""
    entities = "*" if entities is None or len(entities) == 0 else ", ".join(entities)
    sql_string = f"SELECT ROWID, {entities} FROM {table_name};"
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.cursor()
        cursor.execute(sql_string)
        print(cursor.fetchall())


if __name__ == "__main__":
    drop_table(TABLE_NAME)
    create_table(TABLE_NAME, entities=DATA)
    add_entry(
        TABLE_NAME,
        entities=ENTITIES,
        values=["test.txt", "Hello, this is some text."]
    )
    print_table(TABLE_NAME)
