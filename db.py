import sqlite3
from typing import List, Tuple

DB_NAME: str = "files.db"
TABLE_NAME: str = "Files"
DATA: List[Tuple[str, str]] = [("File_name", "TEXT"), ("Contents", "TEXT")]


def sql_execute(
    sql_string: str,
    parameters: tuple = (),
) -> str:
    """Execute provided SQL string with parameters on db."""
    with sqlite3.connect(DB_NAME) as connection:
        cursor = connection.execute(sql_string, parameters)
    return cursor.fetchall()


def create_table(entities: List[Tuple[str, str]]) -> None:
    """Create table if it does not exist already.

    :entities: List of tuples containing (entity_name, entity_format), e.g.
        ("Key", "INT PRIMARY_KEY")

    """
    entity_list = []
    for entity_name, *entity_format in entities:
        entity_list.append(f"[{entity_name}] {', '.join(entity_format)}")
    entity_string = ", ".join(entity_list)

    sql_string = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ({entity_string});"
    sql_execute(sql_string)


def drop_table() -> None:
    """Drop table if it exists."""
    sql_string = f"DROP TABLE IF EXISTS {TABLE_NAME};"
    sql_execute(sql_string)


def add_entry(values: List[str]) -> None:
    """Add entry to existing db table.

    :values: List of values to add.

    """
    sql_string = f"INSERT INTO {TABLE_NAME} VALUES(?, ?);"
    parameters = tuple(values)
    sql_execute(sql_string, parameters)


def get_all_entries() -> List[tuple]:
    """Get all entries from existing db table."""
    sql_string = f"SELECT ROWID, * FROM {TABLE_NAME};"
    return sql_execute(sql_string)


def get_entry_by_id(rowid: int) -> tuple:
    """Get entry from existing db table by id."""
    sql_string = f"SELECT * FROM {TABLE_NAME} WHERE ROWID=?;"
    parameters = (rowid,)
    entry, = sql_execute(sql_string, parameters)
    return entry


def print_table() -> None:
    """Print contents of db table with rowid."""
    sql_string = f"SELECT ROWID, * FROM {TABLE_NAME};"
    print(sql_execute(sql_string))


if __name__ == "__main__":
    drop_table()
    create_table(DATA)
    add_entry(["test.txt", "Hello, this is some text."])
    print_table()
