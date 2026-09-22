"""Natural language questions about IMDB through a local LLM (Ollama) + Ontop.

Flow: question -> the LLM generates SPARQL (with the ontology schema and examples)
-> syntax check -> execution on the Ontop endpoint (retrying on an error or an
empty result) -> the LLM writes the answer from the results.

Usage:
    uv run nl_query.py "Who directed Fargo, from 1996?"
    uv run nl_query.py                 # interactive mode
    uv run nl_query.py --evaluate      # compares against the reference SQL in queries/*.rq
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

# Few-shot examples for the prompt: a fixed pair covering the two patterns the
# ontology describes but does not teach how to use - the n-ary :Performance
# relation (with OPTIONAL and a name literal carrying a suffix) and aggregation
# with COUNT/GROUP BY plus MAX in a subquery. The remaining queries/*.rq stay as
# references for --evaluate, but do not enter the prompt.
FEWSHOT = ("11-cast-with-characters.rq", "15-most-frequent-director.rq")

SPARQL_FORMAT = {
    "type": "object",
    "properties": {"sparql": {"type": "string"}},
    "required": ["sparql"],
}


def schema_summary(ontology: Path) -> str:
    """Summarize the ontology (classes, properties, domain/range) for the prompt."""
    g = Graph().parse(ontology)
    for prefix, ns in PREFIXES.items():
        g.bind(prefix, ns, replace=True)

    def qn(term) -> str:
        return term.n3(g.namespace_manager)

    def comment(term) -> str:
        # Terms carry rdfs:comment in both languages; the prompt takes the English one.
        comments = list(g.objects(term, RDFS.comment))
        english = [c for c in comments if getattr(c, "language", None) == "en"]
        c = next(iter(english or comments), None)
        return f" — {c}" if c else ""

    lines = ["Classes:"]
    for c in sorted(g.subjects(RDF.type, OWL.Class), key=qn):
        parents = ", ".join(qn(p) for p in g.objects(c, RDFS.subClassOf))
        lines.append(f"  {qn(c)}{f' (subclass of {parents})' if parents else ''}{comment(c)}")

    for kind, title in ((OWL.ObjectProperty, "Object properties"), (OWL.DatatypeProperty, "Data properties")):
        lines.append(f"{title}:")
        for p in sorted(g.subjects(RDF.type, kind), key=qn):
            dom = g.value(p, RDFS.domain)
            rng = g.value(p, RDFS.range)
            inv = g.value(p, OWL.inverseOf)
            sig = f"{qn(dom) if dom else '?'} -> {qn(rng) if rng else '?'}"
            if inv:
                sig = f"inverse of {qn(inv)}"
            lines.append(f"  {qn(p)} ({sig}){comment(p)}")
    return "\n".join(lines)


def genre_labels() -> list[str]:
    """Query the endpoint for the existing genre labels, for the prompt."""
    df = sparql(
        "PREFIX : <http://www.example.org/imdb#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "SELECT DISTINCT ?l WHERE { ?g a :Genre ; rdfs:label ?l } ORDER BY ?l"
    )
    return df["l"].to_list()


def strip_comments(query: str) -> str:
    """Drop the header comments of a .rq file, leaving only the SPARQL."""
    return "\n".join(l for l in query.splitlines() if not l.startswith("#")).strip()


def load_examples() -> list[dict]:
    """Load the queries/*.rq files that carry a question in the header."""
    examples = []
    for path in sorted((ROOT / "queries").glob("*.rq")):
        text = path.read_text()
        meta = headers(text)
        if "question" in meta:
            examples.append({"name": path.name, "meta": meta, "sparql": strip_comments(text)})
    return examples


def select_examples(examples: list[dict], exclude: str | None = None) -> list[dict]:
    """Pick the examples that go in the prompt: FEWSHOT, minus `exclude`."""
    by_name = {e["name"]: e for e in examples}
    return [by_name[n] for n in FEWSHOT if n in by_name and n != exclude]


def system_prompt(schema: str, genres: list[str]) -> str:
    """Build the system prompt: schema, vocabulary, rules and query patterns."""
    prefix_lines = "\n".join(f"PREFIX {p}: <{ns}>" for p, ns in PREFIXES.items())
    return f"""You translate questions about films into SPARQL 1.1, run against an Ontop
endpoint (a virtual graph over a relational IMDb database, data up to 2008).
Use ONLY the terms of the ontology below. Answer with JSON {{"sparql": "..."}}.

Prefixes (declare the ones you use):
{prefix_lines}

{schema}

Individuals and literals:
- Films: :title (original title, usually in English) and :releaseYear (integer, e.g. 1994).
  Titles repeat; use :releaseYear whenever the year is mentioned.
- People: first name in foaf:givenName and last name in foaf:familyName (separate).
  Some names carry a suffix, e.g. "Carl (I)". Actors and directors are distinct individuals.
- Genres: :Genre individuals with an rdfs:label in English tagged @en. Existing labels:
  {", ".join(f'"{g}"@en' for g in genres)}.
- Characters: ?p :performer ?actor ; :performanceIn ?film ; :characterName ?name.
- Director-genre affinity with a score: ?af :affinityDirector ?d ; :affinityGenre ?g ; :affinityScore ?s.

Rules:
- SELECT only. Return readable values (names, titles, labels), not just IRIs.
- DO NOT invent individual IRIs (there is no :Drama, :Tarantino etc.). Films, people and
  genres are always variables identified by literals (:title, foaf:givenName/foaf:familyName,
  rdfs:label).
- People do NOT have rdfs:label. Return ?givenName and ?familyName as separate columns
  (do not concatenate them).
- Prefer exact literal equality; use FILTER(CONTAINS(LCASE(STR(?x)), "...")) only if needed.
- Use LIMIT (at most {DEFAULT_LIMIT}) in listings; use ORDER BY when the question asks for a ranking.
- Counts per group: SELECT ?k (COUNT(DISTINCT ?x) AS ?n) ... GROUP BY ?k. To count something
  that may not exist, put the pattern in OPTIONAL. Do not use "AS" outside SELECT/BIND.
- Avoid counting every performance (:Performance) with no filter: it is slow.

Query patterns (adapt the variables):
- Films of a genre:        ?m :hasGenre ?g ; :title ?title . ?g rdfs:label "Comedy"@en .
- Person by name:          ?x foaf:givenName "Name" ; foaf:familyName "Surname" .
- Directors of a film:     ?m :title "Title" ; :hasDirector ?d . ?d foaf:givenName ?givenName ; foaf:familyName ?familyName .
- Films of an actor:       ?a foaf:givenName "Name" ; foaf:familyName "Surname" ; :actedIn ?m . ?m :title ?title .
- Filter by period:        ?m :releaseYear ?year . FILTER(?year >= 1980 && ?year <= 1989)"""


def chat(messages: list[dict], fmt: dict | None = None) -> str:
    """Call the model on Ollama (temperature 0; fmt forces a JSON answer)."""
    resp = ollama.chat(
        model=MODEL,
        messages=messages,
        format=fmt,
        think=False,
        options={"temperature": 0},
    )
    return resp.message.content


def ensure_prefixes_and_limit(query: str) -> str:
    """Normalize the PREFIX declarations and enforce a LIMIT on the generated query.

    Adds the missing prefixes and rewrites those the model declared with a
    divergent IRI - a wrong namespace raises no syntax error, it just makes the
    query match nothing.
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
    """Generate the SPARQL and run it, returning (query, result, attempt count).

    A syntax error, an endpoint error or an empty result go back to the model as
    a new message, up to MAX_ATTEMPTS.
    """
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["meta"]["question"]})
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
            error = f"the endpoint returned an error: {e.read().decode(errors='replace')[:800]}"
        except Exception as e:
            error = f"the query is invalid: {str(e)[:800]}"
        else:
            if result.height > 0 or attempt == MAX_ATTEMPTS:
                return query, result, attempt
            error = (
                "the query returned no results. Check the literals (exact title, first and "
                "last name separate, genre labels tagged @en) and whether the patterns are right."
            )
        if verbose:
            print(f"  attempt {attempt} failed: {error.splitlines()[0]}")
        messages.append({"role": "user", "content": f"Error: {error}\nFix it and answer again in JSON."})
    return query, None, MAX_ATTEMPTS


def parse_json_sparql(raw: str) -> str:
    """Extract the 'sparql' field from the JSON returned by the model."""
    return json.loads(raw)["sparql"]


def answer(question: str, query: str, result: pl.DataFrame | None) -> str:
    """Second call to the model: write the answer using only the rows obtained."""
    if result is None:
        data = "The query failed; there are no results."
    else:
        shown = result.head(ANSWER_MAX_ROWS)
        data = shown.write_csv()
        if result.height > ANSWER_MAX_ROWS:
            data += f"\n(showing {ANSWER_MAX_ROWS} of {result.height} rows)"
        if result.height == 0:
            data = "(no rows returned)"
    return chat(
        [
            {
                "role": "system",
                "content": "You answer questions about films directly and concisely, using "
                "EXCLUSIVELY the results provided. Do not invent data. If there are no "
                "results, say the database does not have that information.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSPARQL executed:\n{query}\n\nResults (CSV):\n{data}",
            },
        ]
    )


def ask(question: str, system: str, examples: list[dict], show_sparql: bool = True):
    """Answer one question and print the SPARQL, a data preview and the answer."""
    t0 = time.perf_counter()
    query, result, attempts = generate_and_run(question, system, examples, verbose=True)
    if show_sparql:
        print(f"\nSPARQL (attempts: {attempts}):\n{query}\n")
        if result is not None:
            print(result.head(10))
    print(f"\nAnswer: {answer(question, query, result)}")
    print(f"({time.perf_counter() - t0:.1f}s)")


def evaluate(system: str, examples: list[dict]):
    """Evaluate the pipeline leave-one-out against the reference SQL of the examples."""
    exact_ok = content_ok = total = 0
    for ex in examples:
        if "sql" not in ex["meta"]:
            continue
        total += 1
        others = select_examples(examples, exclude=ex["name"])  # leave-one-out
        t0 = time.perf_counter()
        query, result, attempts = generate_and_run(ex["meta"]["question"], system, others, verbose=False)
        elapsed = time.perf_counter() - t0
        expected = pl.read_database_uri(ex["meta"]["sql"], POSTGRES_URI)
        exact = result is not None and loose(result) == loose(expected)
        content = exact or (result is not None and content_match(result, expected))
        exact_ok += exact
        content_ok += content
        status = "OK" if exact else "CONTENT" if content else "FAIL"
        got = "error" if result is None else f"{result.height} rows"
        print(f"[{status}] {ex['name']}: {got} vs {expected.height} expected "
              f"({attempts} attempt(s), {elapsed:.1f}s)")
        if not exact:
            print("    " + query.replace("\n", "\n    "))
    print(f"\nModel: {MODEL}")
    print(f"{exact_ok}/{total} identical to the reference")
    print(f"{content_ok}/{total} with correct content (column formatting may differ)")


def loose(df: pl.DataFrame) -> list[tuple]:
    """Multiset of rows, ignoring column names and order."""
    return sorted((tuple(sorted(row, key=repr)) for row in normalize(df)), key=repr)


def _is_number(cell: str) -> bool:
    """Tell whether the cell is numeric (numbers and text are compared differently)."""
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _tokens(cells) -> set[str]:
    """Reduce a row to comparable tokens: normalized numbers and lowercased words."""
    out = set()
    for cell in cells:
        if cell is None:
            continue
        cell = str(cell)
        out |= {format(float(cell), "g")} if _is_number(cell) else set(cell.lower().split())
    return out


def content_match(result: pl.DataFrame, expected: pl.DataFrame) -> bool:
    """Format-tolerant criterion: same row count and a one-to-one pairing in which
    (a) everything the result row contains exists in the reference row and
    (b) all the text (non-numeric) of the reference row appears in the result.
    Accepts first+last name concatenated or numeric columns omitted; rejects
    IRIs in place of names, or different rows."""
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
    parser.add_argument("question", nargs="?")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-sparql", action="store_true", help="do not print the generated SPARQL")
    args = parser.parse_args()

    system = system_prompt(schema_summary(ROOT / "obda" / "imdb-ontology.ttl"), genre_labels())
    examples = load_examples()
    fewshot = select_examples(examples)

    if args.evaluate:
        evaluate(system, examples)
    elif args.question:
        ask(args.question, system, fewshot, not args.no_sparql)
    else:
        print(f"Model: {MODEL}. Ctrl+D to exit.")
        for line in sys.stdin:
            if line.strip():
                ask(line.strip(), system, fewshot, not args.no_sparql)
            print("\n> ", end="", flush=True)


if __name__ == "__main__":
    main()
