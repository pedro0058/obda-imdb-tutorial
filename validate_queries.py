"""Roda as queries SPARQL de queries/*.rq contra o endpoint Ontop e as valida.

Cada arquivo .rq pode trazer, em comentários de cabeçalho:
    # titulo: descrição exibida
    # sql: consulta SQL equivalente no Postgres (gabarito)
    # mostrar-sql: sim   -> exibe a reformulação SQL gerada pelo Ontop

Antes das queries, verifica que os mapeamentos não afirmam rdf:type :Actor
nem :Director — ou seja, que esses tipos só podem vir do rewriting.

Uso:
    docker compose up -d
    uv run validate_queries.py [filtro]
"""

import io
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
from rdflib import RDF, Graph, Namespace

ROOT = Path(__file__).parent
ENDPOINT = os.environ.get("ONTOP_ENDPOINT", "http://localhost:8080")
POSTGRES_URI = os.environ.get(
    "POSTGRES_URI", "postgresql://imdb:imdb@localhost:5432/imdb"
)
R2RML = Namespace("http://www.w3.org/ns/r2rml#")
IMDB = Namespace("http://www.example.org/imdb#")
# Tipos que só podem aparecer por rewriting, nunca afirmados pelo mapeamento.
INFERRED_ONLY = {IMDB.Actor: ":Actor", IMDB.Director: ":Director"}


def check_no_materialized_types(mapping: Path) -> bool:
    """Confere que os mapeamentos não afirmam os tipos que devem vir do rewriting.

    Em R2RML uma classe pode ser afirmada de duas formas: `rr:class` no subjectMap,
    ou um predicateObjectMap com `rr:predicate rdf:type` — as duas são verificadas.
    """
    g = Graph().parse(mapping)
    asserted = set(g.objects(None, R2RML["class"]))
    for pom in g.subjects(R2RML.predicate, RDF.type):
        asserted |= set(g.objects(pom, R2RML.object))
        for om in g.objects(pom, R2RML.objectMap):
            asserted |= set(g.objects(om, R2RML.constant))
    hits = sorted(INFERRED_ONLY[c] for c in asserted & INFERRED_ONLY.keys())
    for h in hits:
        print(f"  tipo materializado encontrado no mapeamento: {h}")
    return not hits


def headers(text: str) -> dict[str, str]:
    """Extrai os metadados dos comentários '# chave: valor' no topo do .rq."""
    return dict(re.findall(r"^#\s*([\w-]+):\s*(.+)$", text, flags=re.MULTILINE))


def http(path: str, params: dict[str, str], accept: str, post: bool = False) -> str:
    """Faz uma requisição HTTP ao endpoint Ontop e devolve o corpo da resposta."""
    data = urllib.parse.urlencode(params)
    if post:
        req = urllib.request.Request(
            f"{ENDPOINT}{path}", data=data.encode(), headers={"Accept": accept}
        )
    else:
        req = urllib.request.Request(f"{ENDPOINT}{path}?{data}", headers={"Accept": accept})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read().decode()


def sparql(query: str) -> pl.DataFrame:
    """Roda uma query SPARQL no endpoint e devolve o resultado como DataFrame."""
    body = http("/sparql", {"query": query}, "text/csv", post=True)
    return pl.read_csv(io.StringIO(body), infer_schema=False)


def normalize(df: pl.DataFrame) -> list[tuple]:
    """Comparação por multiconjunto; números comparados como float."""
    cols = []
    for name in df.columns:
        as_str = df[name].cast(pl.String)
        as_num = as_str.cast(pl.Float64, strict=False)
        cols.append(as_num if as_num.null_count() == as_str.null_count() else as_str)
    return sorted(zip(*cols), key=repr)


def main():
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    pl.Config.set_tbl_rows(30)
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_hide_dataframe_shape(True)

    ok = check_no_materialized_types(ROOT / "obda" / "imdb.r2rml.ttl")
    print(
        f"[{'OK' if ok else 'FALHA'}] mapeamentos sem rdf:type "
        f"{' / '.join(INFERRED_ONLY.values())}\n"
    )

    for path in sorted((ROOT / "queries").glob("*.rq")):
        if filtro not in path.name:
            continue
        text = path.read_text()
        meta = headers(text)
        print(f"=== {path.name}: {meta.get('titulo', '')}")

        t0 = time.perf_counter()
        result = sparql(text)
        elapsed = time.perf_counter() - t0
        print(result)

        if meta.get("mostrar-sql") == "sim":
            reformulation = http("/ontop/reformulate", {"query": text}, "text/plain")
            print("  SQL gerado pelo Ontop:")
            print("    " + reformulation.strip().replace("\n", "\n    "))

        if "sql" in meta:
            expected = pl.read_database_uri(meta["sql"], POSTGRES_URI)
            match = normalize(result) == normalize(expected)
            ok &= match
            print(
                f"[{'OK' if match else 'FALHA'}] SPARQL ({result.height} linhas, "
                f"{elapsed:.1f}s) == SQL gabarito ({expected.height} linhas)"
            )
            if not match:
                print(expected)
        print()

    print("TODAS AS VALIDAÇÕES PASSARAM" if ok else "HÁ VALIDAÇÕES COM FALHA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
