"""Copy the imdb_ijs database (CTU Prague's public MariaDB) into the local Postgres.

Tables, columns, primary and foreign keys are discovered through
information_schema - no table name is hardcoded. Polars is the transport layer:
reading with connectorx, writing with ADBC (binary COPY).

Usage:
    docker compose up -d
    uv run imdb_to_postgres.py
"""

import os
import re
import time

import adbc_driver_postgresql.dbapi as pg
import polars as pl

MARIADB_URI = os.environ.get(
    "MARIADB_URI",
    "mysql://guest:ctu-relational@relational.fel.cvut.cz:3306/imdb_ijs",
)
MARIADB_SCHEMA = os.environ.get("MARIADB_SCHEMA", "imdb_ijs")
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgresql://imdb:imdb@localhost:5432/imdb"
)

# Tables larger than this are read in parallel, partitioned by the PK.
PARTITION_THRESHOLD = 200_000
PARTITION_NUM = 4

# MariaDB type -> (Postgres type, Polars dtype). The dtypes must match the DDL,
# because ADBC's binary COPY does not coerce types.
TYPE_MAP: dict[str, tuple[str, pl.DataType]] = {
    "tinyint": ("smallint", pl.Int16),
    "smallint": ("smallint", pl.Int16),
    "mediumint": ("integer", pl.Int32),
    "int": ("integer", pl.Int32),
    "bigint": ("bigint", pl.Int64),
    "float": ("real", pl.Float32),
    "double": ("double precision", pl.Float64),
    "decimal": ("numeric", pl.Float64),
    "varchar": ("varchar", pl.String),
    "char": ("char", pl.String),
    "text": ("text", pl.String),
    "mediumtext": ("text", pl.String),
    "longtext": ("text", pl.String),
    "date": ("date", pl.Date),
    "datetime": ("timestamp", pl.Datetime),
    "timestamp": ("timestamp", pl.Datetime),
}


def maria(query: str) -> pl.DataFrame:
    """Run a query on the source MariaDB and return the result."""
    return pl.read_database_uri(query, MARIADB_URI)


def discover() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read from the source information_schema: tables, columns and keys (PK/FK)."""
    tables = maria(
        f"""
        SELECT table_name, table_rows
        FROM information_schema.tables
        WHERE table_schema = '{MARIADB_SCHEMA}' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )
    columns = maria(
        f"""
        SELECT table_name, column_name, data_type, character_maximum_length,
               is_nullable, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = '{MARIADB_SCHEMA}'
        ORDER BY table_name, ordinal_position
        """
    )
    keys = maria(
        f"""
        SELECT k.table_name, k.constraint_name, k.column_name, k.ordinal_position,
               k.referenced_table_name, k.referenced_column_name
        FROM information_schema.key_column_usage k
        WHERE k.table_schema = '{MARIADB_SCHEMA}'
        ORDER BY k.table_name, k.constraint_name, k.ordinal_position
        """
    )
    return tables, columns, keys


def quote(ident: str) -> str:
    """Quote an identifier for use in Postgres SQL."""
    return '"' + ident.replace('"', '""') + '"'


def pg_type(row: dict) -> tuple[str, pl.DataType]:
    """Translate a MariaDB column into (Postgres type, Polars dtype)."""
    base, dtype = TYPE_MAP.get(row["data_type"], ("text", pl.String))
    if base in ("varchar", "char") and row["character_maximum_length"]:
        base = f"{base}({int(row['character_maximum_length'])})"
    return base, dtype


def copy_table(conn, table: str, cols: pl.DataFrame, pk_cols: list[str], est_rows: int):
    """Recreate the table on Postgres and transfer its data from the source.

    Large tables are read in parallel, partitioned by the first integer column
    of the PK; writing uses binary COPY through ADBC.
    """
    col_rows = cols.to_dicts()
    ddl_cols = []
    schema: dict[str, pl.DataType] = {}
    for c in col_rows:
        typ, dtype = pg_type(c)
        null = "" if c["is_nullable"] == "YES" else " NOT NULL"
        ddl_cols.append(f"{quote(c['column_name'])} {typ}{null}")
        schema[c["column_name"]] = dtype

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {quote(table)} CASCADE")
        cur.execute(f"CREATE TABLE {quote(table)} ({', '.join(ddl_cols)})")
    conn.commit()

    select_cols = ", ".join(f"`{c['column_name']}`" for c in col_rows)
    query = f"SELECT {select_cols} FROM `{table}`"

    # Partition by the first integer column of the PK, if the table is large.
    partition_on = next(
        (c for c in pk_cols if schema[c] in (pl.Int16, pl.Int32, pl.Int64)), None
    )
    t0 = time.perf_counter()
    if partition_on and est_rows > PARTITION_THRESHOLD:
        df = pl.read_database_uri(
            query,
            MARIADB_URI,
            partition_on=partition_on,
            partition_num=PARTITION_NUM,
        )
    else:
        df = pl.read_database_uri(query, MARIADB_URI)
    t_read = time.perf_counter() - t0

    df = df.select(pl.col(name).cast(dtype) for name, dtype in schema.items())

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.adbc_ingest(table, df.to_arrow(), mode="append")
    conn.commit()
    t_write = time.perf_counter() - t0

    print(
        f"  {table:<20} {df.height:>10,} rows  "
        f"(read {t_read:5.1f}s, write {t_write:5.1f}s)"
    )


def add_constraints(conn, keys: pl.DataFrame):
    """Recreate the primary and foreign keys after the data load."""
    grouped = (
        keys.group_by("table_name", "constraint_name", maintain_order=True)
        .agg(
            pl.col("column_name").sort_by("ordinal_position"),
            pl.col("referenced_table_name").first(),
            pl.col("referenced_column_name").sort_by("ordinal_position"),
        )
        .to_dicts()
    )
    # PKs first, FKs afterwards (FKs depend on the referenced PKs/uniques).
    for g in sorted(grouped, key=lambda g: g["constraint_name"] != "PRIMARY"):
        table, cols = g["table_name"], ", ".join(map(quote, g["column_name"]))
        if g["constraint_name"] == "PRIMARY":
            name = f"{table}_pkey"
            sql = f"ALTER TABLE {quote(table)} ADD CONSTRAINT {quote(name)} PRIMARY KEY ({cols})"
        elif g["referenced_table_name"]:
            name = re.sub(r"\W", "_", f"{table}_{g['constraint_name']}")
            refs = ", ".join(map(quote, g["referenced_column_name"]))
            sql = (
                f"ALTER TABLE {quote(table)} ADD CONSTRAINT {quote(name)} "
                f"FOREIGN KEY ({cols}) REFERENCES {quote(g['referenced_table_name'])} ({refs})"
            )
        else:
            continue  # non-PK UNIQUE indexes: not used in this tutorial
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print(f"  ok    {sql}")
        except Exception as e:  # e.g. an FK violated by orphan rows in the source
            conn.rollback()
            print(f"  FAIL  {sql}\n        {str(e).splitlines()[0]}")


def main():
    print(f"Discovering the schema in {MARIADB_SCHEMA}...")
    tables, columns, keys = discover()
    print(f"  {tables.height} tables: {', '.join(tables['table_name'])}")

    with pg.connect(POSTGRES_URI) as conn:
        print("Copying tables...")
        for t in tables.to_dicts():
            name = t["table_name"]
            pk_cols = (
                keys.filter(
                    (pl.col("table_name") == name)
                    & (pl.col("constraint_name") == "PRIMARY")
                )
                .sort("ordinal_position")["column_name"]
                .to_list()
            )
            copy_table(
                conn,
                name,
                columns.filter(pl.col("table_name") == name),
                pk_cols,
                int(t["table_rows"] or 0),
            )

        print("Creating primary and foreign keys...")
        add_constraints(conn, keys)

        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
