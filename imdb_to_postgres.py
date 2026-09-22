"""Copia o banco imdb_ijs (MariaDB público da CTU Prague) para o Postgres local.

Tabelas, colunas, chaves primárias e estrangeiras são descobertas via
information_schema — nenhum nome de tabela é hardcoded. Polars é a camada de
transporte: leitura com connectorx, escrita com ADBC (COPY binário).

Uso:
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

# Tabelas acima deste tamanho são lidas em paralelo, particionadas pela PK.
PARTITION_THRESHOLD = 200_000
PARTITION_NUM = 4

# tipo MariaDB -> (tipo Postgres, dtype Polars). Os dtypes precisam casar com o
# DDL, pois o COPY binário do ADBC não faz coerção de tipos.
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
    """Executa uma consulta no MariaDB de origem e devolve o resultado."""
    return pl.read_database_uri(query, MARIADB_URI)


def discover() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Lê do information_schema da origem: tabelas, colunas e chaves (PK/FK)."""
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
    """Protege um identificador para uso no SQL do Postgres."""
    return '"' + ident.replace('"', '""') + '"'


def pg_type(row: dict) -> tuple[str, pl.DataType]:
    """Traduz uma coluna do MariaDB para (tipo Postgres, dtype Polars)."""
    base, dtype = TYPE_MAP.get(row["data_type"], ("text", pl.String))
    if base in ("varchar", "char") and row["character_maximum_length"]:
        base = f"{base}({int(row['character_maximum_length'])})"
    return base, dtype


def copy_table(conn, table: str, cols: pl.DataFrame, pk_cols: list[str], est_rows: int):
    """Recria a tabela no Postgres e transfere seus dados da origem.

    Tabelas grandes são lidas em paralelo, particionadas pela primeira coluna
    inteira da PK; a escrita usa COPY binário via ADBC.
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

    # Particiona pela primeira coluna inteira da PK, se a tabela for grande.
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
        f"  {table:<20} {df.height:>10,} linhas  "
        f"(leitura {t_read:5.1f}s, escrita {t_write:5.1f}s)"
    )


def add_constraints(conn, keys: pl.DataFrame):
    """Recria as chaves primárias e estrangeiras depois da carga dos dados."""
    grouped = (
        keys.group_by("table_name", "constraint_name", maintain_order=True)
        .agg(
            pl.col("column_name").sort_by("ordinal_position"),
            pl.col("referenced_table_name").first(),
            pl.col("referenced_column_name").sort_by("ordinal_position"),
        )
        .to_dicts()
    )
    # PKs primeiro, FKs depois (FKs dependem das PKs/uniques referenciadas).
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
            continue  # índices UNIQUE não-PK: não usados neste tutorial
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print(f"  ok    {sql}")
        except Exception as e:  # ex.: FK violada por linhas órfãs na fonte
            conn.rollback()
            print(f"  FALHA {sql}\n        {str(e).splitlines()[0]}")


def main():
    print(f"Descobrindo schema em {MARIADB_SCHEMA}...")
    tables, columns, keys = discover()
    print(f"  {tables.height} tabelas: {', '.join(tables['table_name'])}")

    with pg.connect(POSTGRES_URI) as conn:
        print("Copiando tabelas...")
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

        print("Criando chaves primárias e estrangeiras...")
        add_constraints(conn, keys)

        with conn.cursor() as cur:
            cur.execute("ANALYZE")
        conn.commit()
    print("Concluído.")


if __name__ == "__main__":
    main()
