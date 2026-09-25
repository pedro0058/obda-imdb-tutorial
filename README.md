# OBDA over IMDB with Ontop (+ natural language questions)

A study reproduction of Part 2 of the tutorial *"From Description Logics to (Virtual)
Knowledge Graphs"* (Renata Wassermann and João Lima, USP): a complete
**Ontology-Based Data Access** pipeline over a real relational database, with a local
LLM layer that translates natural language questions into SPARQL.

```
public MariaDB (imdb_ijs) ──Polars──▶ local Postgres ◀──SQL── Ontop ◀──SPARQL── nl_query.py ◀── question
                                                           ▲                        │
                                      obda/imdb-ontology.ttl + obda/imdb.r2rml.ttl   Ollama (gemma4)
```

No triples are materialized: Ontop rewrites every SPARQL query into SQL over Postgres,
using the ontology (OWL 2 QL) and the mappings.

## Layout

| File | Role |
|---|---|
| `docker-compose.yml` | Postgres 16, JDBC driver download and the Ontop 5.5.0 endpoint |
| `imdb_to_postgres.py` | Copies `imdb_ijs` from the public MariaDB into the local Postgres |
| `obda/imdb-ontology.ttl` | OWL 2 QL ontology |
| `obda/imdb.r2rml.ttl` | R2RML mappings (tables → triples) |
| `obda/imdb.properties` | Endpoint's JDBC connection (inside the compose network) |
| `obda/imdb-ontology.obda` | The same mappings in the native format, which Protégé edits |
| `obda/imdb-ontology.properties` | Protégé's JDBC connection (outside Docker, over `localhost`) |
| `queries/*.rq` | Validation SPARQL queries, with the question and the reference SQL in the header |
| `validate_queries.py` | Runs the queries and compares them against the reference SQL |
| `nl_query.py` | Natural language question → SPARQL → answer, via Ollama |

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11)
- Docker with Docker Compose
- [Ollama](https://ollama.com/) with the `gemma4:12b-nvfp4` model (only for the LLM layer)

```bash
uv sync
```

## 1. Infrastructure and data

The source is the [IMDb dataset from CTU Prague](https://relational.fel.cvut.cz/dataset/IMDb)
(public MariaDB, user `guest`). The script discovers tables, columns, primary and foreign
keys through `information_schema` (nothing is hardcoded), creates the equivalent DDL in
Postgres, moves the data with Polars (reading with connectorx, partitioned by the PK on
large tables; writing with ADBC/binary COPY) and recreates PKs and FKs at the end.

```bash
docker compose up -d --wait postgres
uv run imdb_to_postgres.py          # ~2 min, dominated by the network
```

Optional variables: `MARIADB_URI`, `MARIADB_SCHEMA`, `POSTGRES_URI`.

Resulting tables (7 tables, ~5.6 million rows) and the columns used in the mappings:

| Table | Mapped columns | PK | Rows |
|---|---|---|---|
| `actors` | `id`, `first_name`, `last_name` | `id` | 817,718 |
| `directors` | `id`, `first_name`, `last_name` | `id` | 86,880 |
| `movies` | `id`, `name`, `year`, `rank` | `id` | 388,269 |
| `roles` | `actor_id`, `movie_id`, `role` | (actor_id, movie_id, role) | 3,431,966 |
| `movies_directors` | `director_id`, `movie_id` | (director_id, movie_id) | 371,180 |
| `movies_genres` | `movie_id`, `genre` | (movie_id, genre) | 395,119 |
| `directors_genres` | `director_id`, `genre`, `prob` | (director_id, genre) | 156,562 |

Inspection:

```bash
docker exec imdb-postgres psql -U imdb -d imdb -c '\dt'
docker exec imdb-postgres psql -U imdb -d imdb -c '\d roles'
```

## 2. Ontology

`obda/imdb-ontology.ttl` — namespace `http://www.example.org/imdb#` (prefix `imdb`, or `:`).

**Classes:** `:Movie`, `:Actor` and `:Director` (subclasses of `foaf:Person`), `:Genre`,
`:Performance`, `:GenreAffinity`.

**Object properties:** `:actedIn` / `:hasActor` (inverses), `:directed` /
`:hasDirector` (inverses), `:hasGenre`, `:hasGenreAffinity`, `:performer`,
`:performanceIn`, `:affinityDirector`, `:affinityGenre`.

**Data properties:** `:title`, `:releaseYear`, `:rank`, `:characterName`,
`:affinityScore`, plus `foaf:givenName` and `foaf:familyName`.

### Modelling decisions

- **Typing inferred through domain/range (the central decision).** No row in the source
  says "this is an actor" or "this is a director", and the mappings do not assert
  `rdf:type :Actor` or `rdf:type :Director` either. The types come from the axioms:
  `:actedIn rdfs:domain :Actor`, `:performer rdfs:range :Actor`,
  `:directed rdfs:domain :Director`, `:hasGenreAffinity rdfs:domain :Director`,
  `:affinityDirector rdfs:range :Director`. For the query `?x a :Actor`, Ontop rewrites
  the request into `SELECT DISTINCT actor_id FROM roles` — an actor is, by definition,
  someone who acted in at least one movie. The same holds for `:Genre`, `:Performance`
  and `:GenreAffinity`. `foaf:Person` is inferred through the subclass axioms.
- **Types asserted only where the table is the extension of the class:** `movies` →
  `:Movie` and `directors` → `foaf:Person`. The second assertion exists because 569
  directors have neither a movie nor a genre affinity; without it they would have no type
  at all. They are people, but they are not inferred to be `:Director`.
- **FOAF reuse only, without *ontology hijacking*.** The FOAF terms are declared exactly
  as in the FOAF 0.99 specification (`foaf:givenName` with no domain/range;
  `foaf:familyName` with domain `foaf:Person` and range `rdfs:Literal`). We add no axioms
  to third-party terms — the only link is `:Actor`/`:Director rdfs:subClassOf
  foaf:Person`, which is a statement about our own terms. Movies use local terms
  (`:Movie`, `:title`, `:releaseYear`) instead of schema.org, since declaring
  domain/range on `schema:name` would affect any data using that term.
- **Genre as an individual, not a literal** (`data:genre/Drama`, with
  `rdfs:label "Drama"@en`), allowing navigation and a future genre hierarchy.
- **`:hasGenre` ≠ `:hasGenreAffinity`.** "The movie belongs to genre X"
  (`movies_genres`) and "the director has an affinity with X" (`directors_genres`) are
  distinct relations.
- **N-ary relations (the W3C pattern) with a binary shortcut.**
  - `roles` has PK (actor, movie, role): the same actor may play several characters in
    the same movie. Each row becomes a `:Performance` with `:performer`,
    `:performanceIn` and `:characterName`; `:actedIn` remains as the actor → movie
    shortcut.
  - `directors_genres` carries a score (`prob`): each row becomes a `:GenreAffinity` with
    `:affinityDirector`, `:affinityGenre` and `:affinityScore`; `:hasGenreAffinity`
    remains as the shortcut.
- **Datatypes within OWL 2 QL.** `xsd:gYear`, `xsd:float` and `xsd:double` do not belong
  to the profile; hence `:releaseYear` is `xsd:integer` and `:rank` and `:affinityScore`
  are `xsd:decimal`.
- **Documentation good practices** (W3C, OOPS!, Garijo & Poveda-Villalón 2020): ontology
  metadata (`dcterms:title`, `description`, `creator`, `license`, `created`, `source`,
  `owl:versionIRI`, `vann:preferredNamespacePrefix`), `rdfs:label` and `rdfs:comment` in
  Portuguese and English on every local term; classes in CamelCase and properties in
  lowerCamelCase.

### Validating the ontology

```bash
# OWL 2 QL profile
docker run --rm -v "$PWD":/work -w /work obolibrary/robot \
  robot validate-profile --profile QL --input obda/imdb-ontology.ttl
# Consistency (HermiT)
docker run --rm -v "$PWD":/work -w /work obolibrary/robot \
  robot reason --reasoner hermit --input obda/imdb-ontology.ttl
```

Result: *"Ontology and imports closure in profile"* and a consistent ontology.

## 3. R2RML mappings

`obda/imdb.r2rml.ttl` holds 12 mappings in [R2RML](https://www.w3.org/TR/r2rml/), the W3C
recommendation for exposing relational databases as RDF. Each mapping is an
`rr:TriplesMap`: the query selecting the rows goes in `rr:logicalTable` (here always an
`rr:R2RMLView` with `rr:sqlQuery`), the subject IRI template in `rr:subjectMap`, and each
property in an `rr:predicateObjectMap`. Individual IRIs
(base `http://www.example.org/imdb/data/`):

| Individual | Template |
|---|---|
| Actor | `data:actor/{id}` |
| Director | `data:director/{id}` |
| Movie | `data:movie/{id}` |
| Genre | `data:genre/{genre}` |
| Performance | `data:performance/{actor_id}/{movie_id}/{role}` |
| Affinity | `data:director/{director_id}/genre-affinity/{genre}` |

Actors and directors get distinct IRIs because the ids of the two tables are independent
in the source. Literal details: `rank` is rounded to one decimal place and emitted as
`xsd:decimal` (avoiding the floating-point noise of `real`); `prob` becomes
`xsd:decimal`; empty roles (27% of `roles`) produce no `:characterName`; `NULL` values
produce no triples. Special characters in roles are encoded in the IRIs by Ontop
(e.g. `Various%2Flyricist`).

Validating the mappings against the ontology and the database:

```bash
docker compose run --rm ontop-jdbc
docker run --rm -v "$PWD":/work -v "$PWD/jdbc":/opt/ontop/jdbc:ro \
  --network fois_tutorial_default --entrypoint /opt/ontop/ontop ontop/ontop:5.5.0 \
  validate --ontology=/work/obda/imdb-ontology.ttl --mapping=/work/obda/imdb.r2rml.ttl \
  --properties=/work/obda/imdb.properties
```

### Editing in Protégé

The Ontop plugin for Protégé runs outside Docker, so it reaches Postgres through the port
published by compose:

| Field | Value |
|---|---|
| Connection URL | `jdbc:postgresql://localhost:5432/imdb` |
| Username | `imdb` |
| Password | `imdb` |
| Driver class | `org.postgresql.Driver` |
| Driver JAR | `jdbc/postgresql-42.7.13.jar` (included in the repository) |

Register the JAR under *Preferences → JDBC Drivers* before opening the **Ontop Mappings**
tab.

The plugin looks for its working files by the base name of the ontology that is open.
Opening `obda/imdb-ontology.ttl`, it finds `obda/imdb-ontology.obda` on its own (the
mappings, in Ontop's native format, which is what the tab edits) and
`obda/imdb-ontology.properties` (the connection above). New mappings can be created there
and tested in the tab itself.

The endpoint keeps loading `obda/imdb.r2rml.ttl` and `obda/imdb.properties` — R2RML
because it is the W3C recommendation, and a separate connection because, from inside the
compose network, the database is `postgres:5432`, not `localhost`. After editing in
Protégé, propagate to the format the endpoint reads:

```bash
docker run --rm -v "$PWD":/work -v "$PWD/jdbc":/opt/ontop/jdbc:ro \
  --network fois_tutorial_default --entrypoint /opt/ontop/ontop ontop/ontop:5.5.0 \
  mapping to-r2rml -i /work/obda/imdb-ontology.obda -t /work/obda/imdb-ontology.ttl \
  --properties=/work/obda/imdb.properties -o /work/obda/imdb.r2rml.ttl
docker compose restart ontop
```

The reverse path is `mapping to-obda -i <r2rml.ttl> -o <mapping.obda>`. It preserves the
mappings, but generates numeric `mappingId`s (`mapping--549609636`) instead of names —
which is why the `.obda` versioned here is the hand-written one, with readable ids.

## Local SPARQL endpoint

With Postgres already populated:

```bash
docker compose up -d          # postgres + ontop-jdbc (downloads the driver) + ontop
docker logs -f imdb-ontop     # wait for "Ontop has completed the setup"
```

- **Web portal (YASGUI):** <http://localhost:8080/>
- **SPARQL endpoint:** `http://localhost:8080/sparql`
- **SQL generated by Ontop for a query:** `http://localhost:8080/ontop/reformulate?query=...`

The service runs in development mode (`--dev`) with `-Xmx2g` (`ONTOP_JAVA_ARGS` in the
compose file). After editing the ontology or the mappings, apply the changes with
`docker compose restart ontop`.

Example with `curl`:

```bash
curl -s http://localhost:8080/sparql -H 'Accept: text/csv' --data-urlencode 'query=
PREFIX : <http://www.example.org/imdb#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?given ?family WHERE {
  ?m :title "Fargo" ; :releaseYear 1996 ; :hasDirector ?d .
  ?d foaf:givenName ?given ; foaf:familyName ?family .
}'
```

Seeing the SQL rewritten for the inferred typing:

```bash
curl -s -G http://localhost:8080/ontop/reformulate \
  --data-urlencode 'query=PREFIX : <http://www.example.org/imdb#> SELECT ?x WHERE { ?x a :Actor }'
```

## 4. Validation with SPARQL

```bash
uv run validate_queries.py        # all of them
uv run validate_queries.py 01     # filters by file name
```

The script first confirms that the mappings do not assert `rdf:type :Actor` or
`rdf:type :Director` — neither through `rr:class`, nor through an `rr:predicateObjectMap`
with `rr:predicate rdf:type`. It then runs each `queries/*.rq` against the endpoint and
compares the result (as a multiset) with the reference SQL in the `# sql:` header.
Queries carrying `# show-sql: yes` also print Ontop's SQL reformulation.

| # | Query | Exercises | Result |
|---|---|---|---|
| 01 | `?x a :Actor` | domain of `:actedIn`, range of `:performer` | 817,718 |
| 02 | `?x a :Director` | domains of `:directed` and `:hasGenreAffinity` | 86,311 |
| 03 | `?x a foaf:Person` | subclass + asserted directors | 904,598 |
| 04 | `?g a :Genre` | ranges | 21 |
| 05 | `?p a :Performance` | domain on the n-ary node | 3,431,966 |
| 06 | Cast of Pulp Fiction (1994) | `:hasActor` (inverse), `:characterName` | 49 rows |
| 07 | Directors of Fargo (1996) | `:hasDirector` (inverse) | Ethan and Joel Coen |
| 08 | Tarantino's genre affinities | `:GenreAffinity` vs. `:hasGenre` | 8 rows |
| 09 | Top 10 dramas of the 1990s | `:hasGenre`, `:rank`, `FILTER` | 10 rows |
| 10 | Movies directed by Tarantino | `:directed`, lexicographic ordering | 10 rows |
| 11 | Cast of Reservoir Dogs with characters | `OPTIONAL` on `:characterName` | 25 rows |
| 12 | Jackson and Thurman together, with director and year | two `:actedIn` on the same movie | 5 rows |
| 13 | Movies by Pulp Fiction's director featuring Jackson or Thurman | `:hasDirector` + `:directed` chained | 4 rows |
| 14 | Directors between 1990–2005 with Jackson and Thurman | range `FILTER` + `DISTINCT` | 5 rows |
| 15 | Who directed the most movies with Jackson | `COUNT`/`GROUP BY` + `MAX` in a subquery | 2 rows (tie) |
| 16 | Earliest movie with Thurman | `MIN` in a subquery, ties preserved | 1 row |

All of them match the reference SQL. Query 01 is the proof of the rewriting: the
generated SQL is
`SELECT COUNT(DISTINCT actor_id) FROM (SELECT DISTINCT actor_id FROM roles)`.

## 5. Natural language questions (Ollama)

Inference is 100% local through Ollama — no cloud APIs. Default model:
`gemma4:12b-nvfp4`, with *thinking* disabled (`think=False`) and temperature 0.

```bash
ollama pull gemma4:12b-nvfp4
uv run nl_query.py "Who directed Fargo, from 1996?"
uv run nl_query.py                          # interactive mode
uv run nl_query.py --no-sparql "..."        # answer only
uv run nl_query.py --evaluate               # evaluation against the reference SQL
OLLAMA_MODEL=gemma4:e4b-nvfp4 uv run nl_query.py "..."   # another model
```

Flow:

1. The system prompt carries a schema summary **generated from the ontology itself**
   (rdflib, taking the English `rdfs:comment` of each term), the list of genre labels
   queried from the endpoint, rules and query patterns. Two fixed few-shot examples come along (`FEWSHOT` in `nl_query.py`), as
   conversation turns: query 11, which covers the n-ary `:Performance` relation with
   `OPTIONAL`, and query 15, which covers `COUNT`/`GROUP BY` with `MAX` in a subquery.
   The remaining `queries/*.rq` serve as references for `--evaluate`, but do not enter
   the prompt — so adding a validation query does not make every question more expensive.
2. The model returns structured JSON `{"sparql": ...}`; the script normalizes the
   prefixes (adding the missing ones and rewriting those that arrive with a divergent
   IRI, which raise no syntax error but make the query match nothing), enforces
   `LIMIT 100` and validates the syntax with rdflib.
3. The query runs on Ontop. A syntax error, an endpoint error or an empty result goes
   back to the model for correction (up to 3 attempts).
4. A second call writes the answer using **only** the results.

Example:

```
$ uv run nl_query.py "How many horror films were released in 1980?"
SELECT (COUNT(DISTINCT ?m) AS ?count)
WHERE { ?m :hasGenre ?g ; :releaseYear 1980 . ?g rdfs:label "Horror"@en . }
Answer: There were 102 horror films released in 1980.
```

### Evaluation

`--evaluate` performs *leave-one-out*: each question from `queries/*.rq` is answered
without its own query among the examples, and the result is compared with the reference
SQL at two levels:

- **identical:** the same multiset of rows (ignoring column names and order);
- **content:** the same number of rows, paired one-to-one, tolerating formatting (first
  and last name concatenated, numeric columns omitted), but rejecting IRIs in place of
  names or differing values.

Result with `gemma4:12b-nvfp4`: **10/16 identical** to the reference and **11/16 with
correct content**. No generated query was rejected for syntax. Questions solved on the
first attempt take 4 to 44 s each (Apple M4 16 GB, generation plus SPARQL execution;
query 05 alone takes ~27 s in Ontop on a cold cache).

The cases that did not match exactly, and what the model got wrong in each:

| Query | Error |
|---|---|
| 02 | counted through `:hasDirector` instead of `?x a :Director` |
| 08 | aggregate subquery using a variable from the outer pattern, which SPARQL scoping does not propagate |
| 09 | omitted the `?year` column (formatting only: the content matches) |
| 13, 14 | name literal with a stray leading space, `" Thurman"` |
| 15 | `LIMIT 1` over a descending `COUNT`, returning one of the two directors tied at the maximum |

The pattern is clear: the vocabulary part (which class, which property, which direction
of the inverse) comes out right, because it comes from the ontology in the prompt. What
escapes is the shape of the result — which columns to return, how to express "the one
that most…" — and the exact form of the literals, which neither of the two sources in the
prompt describes.

**Limitations:** the evaluation is small (16 questions, several of them structurally
similar to the examples) — it is a smoke test, not a benchmark. Titles are kept in the
original, usually in English and with the article at the end ("Godfather, The"), so
questions using translated titles tend not to find the movie.

## License

- Ontology, mappings, queries and documentation: [CC0 1.0](LICENSE-CC0) (public domain).
- Code: [MIT-0](LICENSE).

The licenses cover only what was created in this repository. **The data is not part of
it:** it is downloaded from the CTU server (originally from kt.ijs.si), derived from IMDb
and subject to its own terms of use. Check them before redistributing dumps or
materialized triples.
