"""Perguntas em linguagem natural sobre o IMDB via LLM local (Ollama) + Ontop.

Fluxo: pergunta -> LLM gera SPARQL (com schema da ontologia e exemplos) ->
checagem de sintaxe -> execução no endpoint Ontop (com nova tentativa em caso
de erro ou resultado vazio) -> LLM redige a resposta a partir dos resultados.

Uso:
    uv run nl_query.py "Quem dirigiu Fargo, de 1996?"
    uv run nl_query.py                 # modo interativo
    uv run nl_query.py --avaliar       # compara com os gabaritos SQL de queries/*.rq
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
from pathlib import Path

import ollama
import polars as pl
from rdflib import OWL, RDF, RDFS, Graph
from rdflib.plugins.sparql import prepareQuery

from validate_queries import POSTGRES_URI, ROOT, headers, normalize, sparql

MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:12b-nvfp4")
MAX_ATTEMPTS = 3
DEFAULT_LIMIT = 100
ANSWER_MAX_ROWS = 60

PREFIXES = {
    "": "http://www.example.org/imdb#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

# Exemplos few-shot do prompt: um par fixo, que cobre os dois padrões que a
# ontologia descreve mas não ensina a usar — a relação n-ária :Performance
# (com OPTIONAL e literal de nome com sufixo) e a agregação com COUNT/GROUP BY
# mais MAX em subconsulta. As demais queries de queries/*.rq continuam servindo
# de gabarito para --avaliar, mas não entram no prompt.
FEWSHOT = ("11-elenco-com-personagens.rq", "15-diretor-mais-frequente.rq")

SPARQL_FORMAT = {
    "type": "object",
    "properties": {"sparql": {"type": "string"}},
    "required": ["sparql"],
}


def schema_summary(ontology: Path) -> str:
    """Resume a ontologia (classes, propriedades, domínio/range) para o prompt."""
    g = Graph().parse(ontology)
    for prefix, ns in PREFIXES.items():
        g.bind(prefix, ns, replace=True)

    def qn(term) -> str:
        return term.n3(g.namespace_manager)

    def comment(term) -> str:
        c = g.value(term, RDFS.comment)
        return f" — {c}" if c else ""

    lines = ["Classes:"]
    for c in sorted(g.subjects(RDF.type, OWL.Class), key=qn):
        parents = ", ".join(qn(p) for p in g.objects(c, RDFS.subClassOf))
        lines.append(f"  {qn(c)}{f' (subclasse de {parents})' if parents else ''}{comment(c)}")

    for kind, title in ((OWL.ObjectProperty, "Propriedades de objeto"), (OWL.DatatypeProperty, "Propriedades de dados")):
        lines.append(f"{title}:")
        for p in sorted(g.subjects(RDF.type, kind), key=qn):
            dom = g.value(p, RDFS.domain)
            rng = g.value(p, RDFS.range)
            inv = g.value(p, OWL.inverseOf)
            sig = f"{qn(dom) if dom else '?'} -> {qn(rng) if rng else '?'}"
            if inv:
                sig = f"inversa de {qn(inv)}"
            lines.append(f"  {qn(p)} ({sig}){comment(p)}")
    return "\n".join(lines)


def genre_labels() -> list[str]:
    """Consulta no endpoint os rótulos de gênero existentes, para o prompt."""
    df = sparql(
        "PREFIX : <http://www.example.org/imdb#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "SELECT DISTINCT ?l WHERE { ?g a :Genre ; rdfs:label ?l } ORDER BY ?l"
    )
    return df["l"].to_list()


def strip_comments(query: str) -> str:
    """Remove os comentários de cabeçalho de um .rq, deixando só o SPARQL."""
    return "\n".join(l for l in query.splitlines() if not l.startswith("#")).strip()


def load_examples() -> list[dict]:
    """Carrega as queries de queries/*.rq que servem de exemplo few-shot."""
    examples = []
    for path in sorted((ROOT / "queries").glob("*.rq")):
        text = path.read_text()
        meta = headers(text)
        if "pergunta" in meta:
            examples.append({"name": path.name, "meta": meta, "sparql": strip_comments(text)})
    return examples


def select_examples(examples: list[dict], exclude: str | None = None) -> list[dict]:
    """Escolhe os exemplos que vão no prompt: FEWSHOT, menos `exclude`."""
    by_name = {e["name"]: e for e in examples}
    return [by_name[n] for n in FEWSHOT if n in by_name and n != exclude]


def system_prompt(schema: str, genres: list[str]) -> str:
    """Monta o prompt de sistema: schema, vocabulário, regras e padrões de consulta."""
    prefix_lines = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in PREFIXES.items())
    return f"""Você traduz perguntas em português sobre filmes para SPARQL 1.1, executado em um
endpoint Ontop (grafo virtual sobre um banco relacional do IMDb, dados até 2008).
Use SOMENTE os termos da ontologia abaixo. Responda com JSON {{"sparql": "..."}}.

Prefixos (declare os que usar):
{prefix_lines}

{schema}

Indivíduos e literais:
- Filmes: :title (título original, geralmente em inglês) e :releaseYear (inteiro, ex.: 1994).
  Há títulos repetidos; use :releaseYear quando o ano for citado.
- Pessoas: nome em foaf:givenName e sobrenome em foaf:familyName (separados).
  Alguns nomes têm sufixo, ex.: "Carl (I)". Atores e diretores são indivíduos distintos.
- Gêneros: indivíduos :Genre com rdfs:label em inglês com tag @en. Rótulos existentes:
  {", ".join(f'"{g}"@en' for g in genres)}.
- Personagens: ?p :performer ?ator ; :performanceIn ?filme ; :characterName ?nome.
- Afinidade diretor-gênero com escore: ?af :affinityDirector ?d ; :affinityGenre ?g ; :affinityScore ?s.

Regras:
- Apenas SELECT. Retorne valores legíveis (nomes, títulos, rótulos), não só IRIs.
- NÃO invente IRIs de indivíduos (não existe :Drama, :Tarantino etc.). Filmes, pessoas e
  gêneros são sempre variáveis identificadas por literais (:title, foaf:givenName/foaf:familyName,
  rdfs:label).
- Pessoas NÃO têm rdfs:label. Retorne ?givenName e ?familyName como colunas separadas
  (não concatene).
- Prefira igualdade exata de literais; use FILTER(CONTAINS(LCASE(STR(?x)), "...")) só se necessário.
- Use LIMIT (no máximo {DEFAULT_LIMIT}) em listagens; use ORDER BY quando a pergunta pedir ranking.
- Contagens por grupo: SELECT ?k (COUNT(DISTINCT ?x) AS ?n) ... GROUP BY ?k. Para contar algo
  que pode não existir, coloque o padrão em OPTIONAL. Não use "AS" fora de SELECT/BIND.
- Evite contar todas as atuações (:Performance) sem filtro: é lento.

Padrões de consulta (adapte as variáveis):
- Filmes de um gênero:   ?m :hasGenre ?g ; :title ?title . ?g rdfs:label "Comedy"@en .
- Pessoa pelo nome:      ?x foaf:givenName "Nome" ; foaf:familyName "Sobrenome" .
- Diretores de um filme: ?m :title "Título" ; :hasDirector ?d . ?d foaf:givenName ?givenName ; foaf:familyName ?familyName .
- Filmes de um ator:     ?a foaf:givenName "Nome" ; foaf:familyName "Sobrenome" ; :actedIn ?m . ?m :title ?title .
- Filtro por período:    ?m :releaseYear ?year . FILTER(?year >= 1980 && ?year <= 1989)"""


def chat(messages: list[dict], fmt: dict | None = None) -> str:
    """Chama o modelo no Ollama (temperatura 0; fmt força a resposta em JSON)."""
    resp = ollama.chat(
        model=MODEL,
        messages=messages,
        format=fmt,
        think=False,
        options={"temperature": 0},
    )
    return resp.message.content


def ensure_prefixes_and_limit(query: str) -> str:
    """Normaliza os PREFIX e impõe um LIMIT à query gerada.

    Acrescenta os prefixos ausentes e reescreve os que o modelo declarou com
    IRI divergente — um namespace errado não dá erro de sintaxe, só faz a query
    não casar com nada.
    """

    def fix(match: re.Match) -> str:
        prefix, iri = match.group(1), match.group(2)
        if prefix in PREFIXES and iri != PREFIXES[prefix]:
            return f"PREFIX {prefix}: <{PREFIXES[prefix]}>"
        return match.group(0)

    query = re.sub(r"PREFIX\s+(\w*):\s*<([^>]*)>", fix, query, flags=re.IGNORECASE)
    declared = set(re.findall(r"PREFIX\s+(\w*):", query, flags=re.IGNORECASE))
    missing = [f"PREFIX {p}: <{ns}>" for p, ns in PREFIXES.items() if p not in declared]
    if not re.search(r"\bLIMIT\s+\d+", query, flags=re.IGNORECASE):
        query = f"{query.rstrip()}\nLIMIT {DEFAULT_LIMIT}"
    return "\n".join(missing + [query])


def generate_and_run(question: str, system: str, examples: list[dict], verbose: bool):
    """Gera o SPARQL e o executa, devolvendo (query, resultado, nº de tentativas).

    Erro de sintaxe, erro do endpoint ou resultado vazio voltam ao modelo como
    nova mensagem, até MAX_ATTEMPTS.
    """
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["meta"]["pergunta"]})
        messages.append({"role": "assistant", "content": json.dumps({"sparql": ex["sparql"]}, ensure_ascii=False)})
    messages.append({"role": "user", "content": question})

    query, result, error = "", None, ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = chat(messages, fmt=SPARQL_FORMAT)
        messages.append({"role": "assistant", "content": raw})
        try:
            query = ensure_prefixes_and_limit(parse_json_sparql(raw))
            prepareQuery(query)
            result = sparql(query)
        except urllib.error.HTTPError as e:
            error = f"o endpoint retornou erro: {e.read().decode(errors='replace')[:800]}"
        except Exception as e:
            error = f"a query é inválida: {str(e)[:800]}"
        else:
            if result.height > 0 or attempt == MAX_ATTEMPTS:
                return query, result, attempt
            error = (
                "a query não retornou resultados. Verifique literais (título exato, nome e "
                "sobrenome separados, rótulos de gênero com @en) e se os padrões estão corretos."
            )
        if verbose:
            print(f"  tentativa {attempt} falhou: {error.splitlines()[0]}")
        messages.append({"role": "user", "content": f"Erro: {error}\nCorrija e responda de novo em JSON."})
    return query, None, MAX_ATTEMPTS


def parse_json_sparql(raw: str) -> str:
    """Extrai o campo 'sparql' do JSON devolvido pelo modelo."""
    return json.loads(raw)["sparql"]


def answer(question: str, query: str, result: pl.DataFrame | None) -> str:
    """Segunda chamada ao modelo: redige a resposta usando só as linhas obtidas."""
    if result is None:
        data = "A consulta falhou; não há resultados."
    else:
        shown = result.head(ANSWER_MAX_ROWS)
        data = shown.write_csv()
        if result.height > ANSWER_MAX_ROWS:
            data += f"\n(mostrando {ANSWER_MAX_ROWS} de {result.height} linhas)"
        if result.height == 0:
            data = "(nenhuma linha retornada)"
    return chat(
        [
            {
                "role": "system",
                "content": "Você responde perguntas sobre filmes em português, de forma direta e "
                "concisa, usando EXCLUSIVAMENTE os resultados fornecidos. Não invente dados. "
                "Se não houver resultados, diga que a base não tem essa informação.",
            },
            {
                "role": "user",
                "content": f"Pergunta: {question}\n\nSPARQL executado:\n{query}\n\nResultados (CSV):\n{data}",
            },
        ]
    )


def ask(question: str, system: str, examples: list[dict], show_sparql: bool = True):
    """Responde uma pergunta e imprime SPARQL, prévia dos dados e resposta."""
    t0 = time.perf_counter()
    query, result, attempts = generate_and_run(question, system, examples, verbose=True)
    if show_sparql:
        print(f"\nSPARQL (tentativas: {attempts}):\n{query}\n")
        if result is not None:
            print(result.head(10))
    print(f"\nResposta: {answer(question, query, result)}")
    print(f"({time.perf_counter() - t0:.1f}s)")


def evaluate(system: str, examples: list[dict]):
    """Avalia o pipeline em leave-one-out contra os gabaritos SQL dos exemplos."""
    exact_ok = content_ok = total = 0
    for ex in examples:
        if "sql" not in ex["meta"]:
            continue
        total += 1
        others = select_examples(examples, exclude=ex["name"])  # leave-one-out
        t0 = time.perf_counter()
        query, result, attempts = generate_and_run(ex["meta"]["pergunta"], system, others, verbose=False)
        elapsed = time.perf_counter() - t0
        expected = pl.read_database_uri(ex["meta"]["sql"], POSTGRES_URI)
        exact = result is not None and loose(result) == loose(expected)
        content = exact or (result is not None and content_match(result, expected))
        exact_ok += exact
        content_ok += content
        status = "OK" if exact else "CONTEÚDO" if content else "FALHA"
        got = "erro" if result is None else f"{result.height} linhas"
        print(f"[{status}] {ex['name']}: {got} vs {expected.height} esperadas "
              f"({attempts} tentativa(s), {elapsed:.1f}s)")
        if not exact:
            print("    " + query.replace("\n", "\n    "))
    print(f"\nModelo: {MODEL}")
    print(f"{exact_ok}/{total} com resultado idêntico ao gabarito")
    print(f"{content_ok}/{total} com conteúdo correto (formato de colunas pode diferir)")


def loose(df: pl.DataFrame) -> list[tuple]:
    """Multiconjunto de linhas, ignorando nomes e ordem das colunas."""
    return sorted((tuple(sorted(row, key=repr)) for row in normalize(df)), key=repr)


def _is_number(cell: str) -> bool:
    """Diz se a célula é numérica (para comparar números e texto de formas diferentes)."""
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _tokens(cells) -> set[str]:
    """Reduz uma linha a tokens comparáveis: números normalizados e palavras minúsculas."""
    out = set()
    for cell in cells:
        if cell is None:
            continue
        cell = str(cell)
        out |= {format(float(cell), "g")} if _is_number(cell) else set(cell.lower().split())
    return out


def content_match(result: pl.DataFrame, expected: pl.DataFrame) -> bool:
    """Critério tolerante a formato: mesmo nº de linhas e pareamento 1-a-1 em que
    (a) tudo o que a linha do resultado contém existe na linha do gabarito e
    (b) todo o texto (não numérico) da linha do gabarito aparece no resultado.
    Aceita nome+sobrenome concatenados ou colunas numéricas omitidas; rejeita
    IRIs no lugar de nomes ou linhas diferentes."""
    if result.height != expected.height:
        return False
    got = [_tokens(r) for r in result.iter_rows()]
    used = [False] * len(got)
    for row in expected.iter_rows():
        all_tokens = _tokens(row)
        text_tokens = _tokens(c for c in row if c is not None and not _is_number(str(c)))
        for i, toks in enumerate(got):
            if not used[i] and toks and toks <= all_tokens and text_tokens <= toks:
                used[i] = True
                break
        else:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pergunta", nargs="?")
    parser.add_argument("--avaliar", action="store_true")
    parser.add_argument("--sem-sparql", action="store_true", help="não exibe o SPARQL gerado")
    args = parser.parse_args()

    system = system_prompt(schema_summary(ROOT / "obda" / "imdb-ontology.ttl"), genre_labels())
    examples = load_examples()
    fewshot = select_examples(examples)

    if args.avaliar:
        evaluate(system, examples)
    elif args.pergunta:
        ask(args.pergunta, system, fewshot, not args.sem_sparql)
    else:
        print(f"Modelo: {MODEL}. Ctrl+D para sair.")
        for line in sys.stdin:
            if line.strip():
                ask(line.strip(), system, fewshot, not args.sem_sparql)
            print("\n> ", end="", flush=True)


if __name__ == "__main__":
    main()
