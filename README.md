# OBDA sobre o IMDB com Ontop (+ perguntas em linguagem natural)

Reprodução, como projeto de estudo, da Parte 2 do tutorial *"From Description Logics
to (Virtual) Knowledge Graphs"* (Renata Wassermann e João Lima, USP): um pipeline
completo de **Ontology-Based Data Access** sobre um banco relacional real, com uma
camada de LLM local que traduz perguntas em português para SPARQL.

```
MariaDB público (imdb_ijs) ──Polars──▶ Postgres local ◀──SQL── Ontop ◀──SPARQL── nl_query.py ◀── pergunta
                                                            ▲                        │
                                       obda/imdb-ontology.ttl + obda/imdb.r2rml.ttl   Ollama (gemma4)
```

Nenhuma tripla é materializada: o Ontop reescreve cada consulta SPARQL em SQL sobre o
Postgres, usando a ontologia (OWL 2 QL) e os mapeamentos.

## Estrutura

| Arquivo | Papel |
|---|---|
| `docker-compose.yml` | Postgres 16, download do driver JDBC e endpoint Ontop 5.5.0 |
| `imdb_to_postgres.py` | Copia o `imdb_ijs` do MariaDB público para o Postgres local |
| `obda/imdb-ontology.ttl` | Ontologia OWL 2 QL |
| `obda/imdb.r2rml.ttl` | Mapeamentos R2RML (tabelas → triplas) |
| `obda/imdb.properties` | Conexão JDBC do endpoint (dentro da rede do compose) |
| `obda/imdb-ontology.obda` | Mesmos mapeamentos no formato nativo, que o Protégé edita |
| `obda/imdb-ontology.properties` | Conexão JDBC do Protégé (fora do Docker, via `localhost`) |
| `queries/*.rq` | Queries SPARQL de validação, com pergunta e SQL gabarito no cabeçalho |
| `validate_queries.py` | Roda as queries e compara com o gabarito SQL |
| `nl_query.py` | Pergunta em linguagem natural → SPARQL → resposta, via Ollama |

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) (Python ≥ 3.11)
- Docker com Docker Compose
- [Ollama](https://ollama.com/) com o modelo `gemma4:12b-nvfp4` (só para a camada de LLM)

```bash
uv sync
```

## 1. Infraestrutura e dados

A fonte é o dataset [IMDb da CTU Prague](https://relational.fel.cvut.cz/dataset/IMDb)
(MariaDB público, usuário `guest`). O script descobre tabelas, colunas, chaves primárias
e estrangeiras via `information_schema` (nada é hardcoded), cria o DDL equivalente no
Postgres, transporta os dados com Polars (leitura com connectorx, particionada pela PK
nas tabelas grandes; escrita com ADBC/COPY binário) e recria PKs e FKs ao final.

```bash
docker compose up -d --wait postgres
uv run imdb_to_postgres.py          # ~2 min, dominado pela rede
```

Variáveis opcionais: `MARIADB_URI`, `MARIADB_SCHEMA`, `POSTGRES_URI`.

Tabelas resultantes (7 tabelas, ~5,6 milhões de linhas) e colunas usadas nos mapeamentos:

| Tabela | Colunas mapeadas | PK | Linhas |
|---|---|---|---|
| `actors` | `id`, `first_name`, `last_name` | `id` | 817.718 |
| `directors` | `id`, `first_name`, `last_name` | `id` | 86.880 |
| `movies` | `id`, `name`, `year`, `rank` | `id` | 388.269 |
| `roles` | `actor_id`, `movie_id`, `role` | (actor_id, movie_id, role) | 3.431.966 |
| `movies_directors` | `director_id`, `movie_id` | (director_id, movie_id) | 371.180 |
| `movies_genres` | `movie_id`, `genre` | (movie_id, genre) | 395.119 |
| `directors_genres` | `director_id`, `genre`, `prob` | (director_id, genre) | 156.562 |

Inspeção:

```bash
docker exec imdb-postgres psql -U imdb -d imdb -c '\dt'
docker exec imdb-postgres psql -U imdb -d imdb -c '\d roles'
```

## 2. Ontologia

`obda/imdb-ontology.ttl` — namespace `http://www.example.org/imdb#` (prefixo `imdb`, ou `:`).

**Classes:** `:Movie`, `:Actor` e `:Director` (subclasses de `foaf:Person`), `:Genre`,
`:Performance`, `:GenreAffinity`.

**Propriedades de objeto:** `:actedIn` / `:hasActor` (inversas), `:directed` /
`:hasDirector` (inversas), `:hasGenre`, `:hasGenreAffinity`, `:performer`,
`:performanceIn`, `:affinityDirector`, `:affinityGenre`.

**Propriedades de dados:** `:title`, `:releaseYear`, `:rank`, `:characterName`,
`:affinityScore`, além de `foaf:givenName` e `foaf:familyName`.

### Decisões de modelagem

- **Tipagem inferida via domínio/range (decisão central).** Nenhuma linha da fonte diz
  "isto é um ator" ou "isto é um diretor", e os mapeamentos também não afirmam
  `rdf:type :Actor` nem `rdf:type :Director`. Os tipos vêm dos axiomas:
  `:actedIn rdfs:domain :Actor`, `:performer rdfs:range :Actor`,
  `:directed rdfs:domain :Director`, `:hasGenreAffinity rdfs:domain :Director`,
  `:affinityDirector rdfs:range :Director`. Na consulta `?x a :Actor`, o Ontop
  reescreve o pedido em `SELECT DISTINCT actor_id FROM roles` — um ator é, por
  definição, alguém que atuou em pelo menos um filme. O mesmo vale para `:Genre`,
  `:Performance` e `:GenreAffinity`. `foaf:Person` é inferido por subclasse.
- **Tipos afirmados só onde a tabela é a extensão da classe:** `movies` → `:Movie` e
  `directors` → `foaf:Person`. A segunda afirmação existe porque 569 diretores não têm
  filme nem afinidade de gênero; sem ela não teriam tipo algum. Eles são pessoas, mas
  não são inferidos `:Director`.
- **Reuso apenas do FOAF, sem *ontology hijacking*.** Os termos FOAF são declarados
  exatamente como na especificação FOAF 0.99 (`foaf:givenName` sem domínio/range;
  `foaf:familyName` com domínio `foaf:Person` e range `rdfs:Literal`). Não adicionamos
  axiomas a termos de terceiros — o único vínculo é `:Actor`/`:Director
  rdfs:subClassOf foaf:Person`, que é uma afirmação sobre termos nossos. Filmes usam
  termos locais (`:Movie`, `:title`, `:releaseYear`) em vez de schema.org, pois declarar
  domínio/range em `schema:name` afetaria qualquer dado que use esse termo.
- **Gênero como indivíduo, não literal** (`data:genre/Drama`, com `rdfs:label "Drama"@en`),
  permitindo navegação e uma futura hierarquia de gêneros.
- **`:hasGenre` ≠ `:hasGenreAffinity`.** "O filme é do gênero X" (`movies_genres`) e
  "o diretor tem afinidade com X" (`directors_genres`) são relações distintas.
- **Relações n-árias (padrão W3C) com atalho binário.**
  - `roles` tem PK (ator, filme, papel): o mesmo ator pode interpretar vários personagens
    no mesmo filme. Cada linha vira uma `:Performance` com `:performer`,
    `:performanceIn` e `:characterName`; `:actedIn` continua como atalho ator → filme.
  - `directors_genres` tem um escore (`prob`): cada linha vira uma `:GenreAffinity` com
    `:affinityDirector`, `:affinityGenre` e `:affinityScore`; `:hasGenreAffinity`
    continua como atalho.
- **Datatypes dentro do OWL 2 QL.** `xsd:gYear`, `xsd:float` e `xsd:double` não
  pertencem ao perfil; por isso `:releaseYear` é `xsd:integer` e `:rank` e
  `:affinityScore` são `xsd:decimal`.
- **Boas práticas de documentação** (W3C, OOPS!, Garijo & Poveda-Villalón 2020):
  metadados da ontologia (`dcterms:title`, `description`, `creator`, `license`,
  `created`, `source`, `owl:versionIRI`, `vann:preferredNamespacePrefix`), rótulos
  `rdfs:label` em pt e en e `rdfs:comment` em todos os termos locais; classes em
  CamelCase e propriedades em lowerCamelCase.

### Validação da ontologia

```bash
# Perfil OWL 2 QL
docker run --rm -v "$PWD":/work -w /work obolibrary/robot \
  robot validate-profile --profile QL --input obda/imdb-ontology.ttl
# Consistência (HermiT)
docker run --rm -v "$PWD":/work -w /work obolibrary/robot \
  robot reason --reasoner hermit --input obda/imdb-ontology.ttl
```

Resultado: *"Ontology and imports closure in profile"* e ontologia consistente.

## 3. Mapeamentos R2RML

`obda/imdb.r2rml.ttl` contém 12 mapeamentos em [R2RML](https://www.w3.org/TR/r2rml/), a
recomendação do W3C para expor bancos relacionais como RDF. Cada mapeamento é um
`rr:TriplesMap`: a consulta que seleciona as linhas fica em `rr:logicalTable` (aqui sempre
um `rr:R2RMLView` com `rr:sqlQuery`), o template da IRI do sujeito em `rr:subjectMap`, e
cada propriedade em um `rr:predicateObjectMap`. IRIs dos indivíduos
(base `http://www.example.org/imdb/data/`):

| Indivíduo | Template |
|---|---|
| Ator | `data:actor/{id}` |
| Diretor | `data:director/{id}` |
| Filme | `data:movie/{id}` |
| Gênero | `data:genre/{genre}` |
| Atuação | `data:performance/{actor_id}/{movie_id}/{role}` |
| Afinidade | `data:director/{director_id}/genre-affinity/{genre}` |

Atores e diretores têm IRIs distintas porque os ids das duas tabelas são independentes na
fonte. Detalhes de literais: `rank` é arredondado para 1 casa e emitido como
`xsd:decimal` (evita ruído de ponto flutuante do `real`); `prob` vira `xsd:decimal`;
papéis vazios (27% de `roles`) não geram `:characterName`; valores `NULL` não geram
triplas. Caracteres especiais nos papéis são codificados nas IRIs pelo Ontop
(ex.: `Various%2Flyricist`).

Validação dos mapeamentos contra a ontologia e o banco:

```bash
docker compose run --rm ontop-jdbc
docker run --rm -v "$PWD":/work -v "$PWD/jdbc":/opt/ontop/jdbc:ro \
  --network fois_tutorial_default --entrypoint /opt/ontop/ontop ontop/ontop:5.5.0 \
  validate --ontology=/work/obda/imdb-ontology.ttl --mapping=/work/obda/imdb.r2rml.ttl \
  --properties=/work/obda/imdb.properties
```

### Editando no Protégé

O plugin Ontop do Protégé roda fora do Docker, então alcança o Postgres pela porta
publicada pelo compose:

| Campo | Valor |
|---|---|
| Connection URL | `jdbc:postgresql://localhost:5432/imdb` |
| Username | `imdb` |
| Password | `imdb` |
| Driver class | `org.postgresql.Driver` |
| Driver JAR | `jdbc/postgresql-42.7.13.jar` (baixado por `docker compose up`) |

Registre o JAR em *Preferences → JDBC Drivers* antes de abrir a aba **Ontop Mappings**.

O plugin procura os arquivos de trabalho pelo nome-base da ontologia aberta. Abrindo
`obda/imdb-ontology.ttl`, ele encontra sozinho `obda/imdb-ontology.obda` (os mapeamentos,
no formato nativo do Ontop, que é o que a aba edita) e `obda/imdb-ontology.properties`
(a conexão acima). Dá para criar mapeamentos novos ali e testá-los na própria aba.

O endpoint continua carregando `obda/imdb.r2rml.ttl` e `obda/imdb.properties` — R2RML
porque é a recomendação do W3C, e uma conexão separada porque, de dentro da rede do
compose, o banco é `postgres:5432`, não `localhost`. Depois de editar no Protégé, propague
para o formato que o endpoint lê:

```bash
docker run --rm -v "$PWD":/work -v "$PWD/jdbc":/opt/ontop/jdbc:ro \
  --network fois_tutorial_default --entrypoint /opt/ontop/ontop ontop/ontop:5.5.0 \
  mapping to-r2rml -i /work/obda/imdb-ontology.obda -t /work/obda/imdb-ontology.ttl \
  --properties=/work/obda/imdb.properties -o /work/obda/imdb.r2rml.ttl
docker compose restart ontop
```

O caminho inverso é `mapping to-obda -i <r2rml.ttl> -o <mapping.obda>`. Ele preserva os
mapeamentos, mas gera `mappingId` numéricos (`mapping--549609636`) no lugar dos nomes —
por isso o `.obda` versionado aqui é o escrito à mão, com ids legíveis.

## Endpoint SPARQL local

Com o Postgres já populado:

```bash
docker compose up -d          # postgres + ontop-jdbc (baixa o driver) + ontop
docker logs -f imdb-ontop     # aguarde "Ontop has completed the setup"
```

- **Portal web (YASGUI):** <http://localhost:8080/>
- **Endpoint SPARQL:** `http://localhost:8080/sparql`
- **SQL gerado pelo Ontop para uma query:** `http://localhost:8080/ontop/reformulate?query=...`

O serviço roda em modo de desenvolvimento (`--dev`) com `-Xmx2g` (`ONTOP_JAVA_ARGS` no
compose). Depois de editar a ontologia ou os mapeamentos, aplique com
`docker compose restart ontop`.

Exemplo com `curl`:

```bash
curl -s http://localhost:8080/sparql -H 'Accept: text/csv' --data-urlencode 'query=
PREFIX : <http://www.example.org/imdb#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?given ?family WHERE {
  ?m :title "Fargo" ; :releaseYear 1996 ; :hasDirector ?d .
  ?d foaf:givenName ?given ; foaf:familyName ?family .
}'
```

Ver o SQL reescrito para a tipagem inferida:

```bash
curl -s -G http://localhost:8080/ontop/reformulate \
  --data-urlencode 'query=PREFIX : <http://www.example.org/imdb#> SELECT ?x WHERE { ?x a :Actor }'
```

## 4. Validação com SPARQL

```bash
uv run validate_queries.py        # todas
uv run validate_queries.py 01     # filtra pelo nome do arquivo
```

O script primeiro confirma que os mapeamentos não afirmam `rdf:type :Actor` nem
`rdf:type :Director` — nem por `rr:class`, nem por um `rr:predicateObjectMap` com
`rr:predicate rdf:type`. Depois roda cada `queries/*.rq` no endpoint e compara o
resultado (como multiconjunto) com o SQL gabarito do cabeçalho `# sql:`. Queries com
`# mostrar-sql: sim` exibem também a reformulação SQL do Ontop.

| # | Query | Exercita | Resultado |
|---|---|---|---|
| 01 | `?x a :Actor` | domínio de `:actedIn`, range de `:performer` | 817.718 |
| 02 | `?x a :Director` | domínios de `:directed` e `:hasGenreAffinity` | 86.311 |
| 03 | `?x a foaf:Person` | subclasse + diretores afirmados | 904.598 |
| 04 | `?g a :Genre` | ranges | 21 |
| 05 | `?p a :Performance` | domínio no nó n-ário | 3.431.966 |
| 06 | Elenco de Pulp Fiction (1994) | `:hasActor` (inversa), `:characterName` | 49 linhas |
| 07 | Diretores de Fargo (1996) | `:hasDirector` (inversa) | Ethan e Joel Coen |
| 08 | Afinidades de gênero de Tarantino | `:GenreAffinity` vs. `:hasGenre` | 8 linhas |
| 09 | Top 10 dramas dos anos 90 | `:hasGenre`, `:rank`, `FILTER` | 10 linhas |
| 10 | Filmes dirigidos por Tarantino | `:directed`, ordenação lexicográfica | 10 linhas |
| 11 | Elenco de Reservoir Dogs com personagens | `OPTIONAL` em `:characterName` | 25 linhas |
| 12 | Jackson e Thurman juntos, com diretor e ano | dois `:actedIn` no mesmo filme | 5 linhas |
| 13 | Filmes do diretor de Pulp Fiction com Jackson ou Thurman | `:hasDirector` + `:directed` encadeados | 4 linhas |
| 14 | Diretores de 1990–2005 com Jackson e Thurman | `FILTER` de intervalo + `DISTINCT` | 5 linhas |
| 15 | Quem mais dirigiu filmes com Jackson | `COUNT`/`GROUP BY` + `MAX` em subconsulta | 2 linhas (empate) |
| 16 | Filme mais antigo com Thurman | `MIN` em subconsulta, empates preservados | 1 linha |

Todas batem com o gabarito SQL. A 01 é a prova do rewriting: SQL gerado
`SELECT COUNT(DISTINCT actor_id) FROM (SELECT DISTINCT actor_id FROM roles)`.

## 5. Perguntas em linguagem natural (Ollama)

Inferência 100% local via Ollama — sem APIs de nuvem. Modelo padrão:
`gemma4:12b-nvfp4`, com *thinking* desligado (`think=False`) e temperatura 0.

```bash
ollama pull gemma4:12b-nvfp4
uv run nl_query.py "Quem dirigiu Fargo, de 1996?"
uv run nl_query.py                          # modo interativo
uv run nl_query.py --sem-sparql "..."       # só a resposta
uv run nl_query.py --avaliar                # avaliação contra os gabaritos
OLLAMA_MODEL=gemma4:e4b-nvfp4 uv run nl_query.py "..."   # outro modelo
```

Fluxo:

1. O prompt de sistema traz um resumo do schema **gerado da própria ontologia** (rdflib),
   a lista de rótulos de gênero consultada no endpoint, regras e padrões de consulta.
   Acompanham dois exemplos few-shot fixos (`FEWSHOT` em `nl_query.py`), como turnos de
   conversa: a query 11, que cobre a relação n-ária `:Performance` com `OPTIONAL`, e a 15,
   que cobre `COUNT`/`GROUP BY` com `MAX` em subconsulta. As demais `queries/*.rq` servem
   de gabarito para `--avaliar`, mas não entram no prompt — assim acrescentar uma query de
   validação não encarece cada pergunta.
2. O modelo devolve JSON estruturado `{"sparql": ...}`; o script normaliza os prefixos
   (acrescenta os ausentes e reescreve os que vierem com IRI divergente, que não dão erro
   de sintaxe mas fazem a query não casar com nada), impõe `LIMIT 100` e valida a sintaxe
   com rdflib.
3. A query roda no Ontop. Erro de sintaxe, erro do endpoint ou resultado vazio voltam ao
   modelo para correção (até 3 tentativas).
4. Uma segunda chamada redige a resposta em português usando **apenas** os resultados.

Exemplo:

```
$ uv run nl_query.py "Quantos filmes de terror foram lançados em 1980?"
SELECT (COUNT(DISTINCT ?m) AS ?n)
WHERE { ?m :hasGenre ?g ; :releaseYear 1980 . ?g rdfs:label "Horror"@en . }
Resposta: Foram lançados 102 filmes de terror em 1980.
```

### Avaliação

`--avaliar` faz *leave-one-out*: cada pergunta de `queries/*.rq` é respondida sem a
própria query entre os exemplos, e o resultado é comparado ao gabarito SQL em dois níveis:

- **idêntico:** mesmo multiconjunto de linhas (ignorando nome e ordem das colunas);
- **conteúdo:** mesmo número de linhas, pareadas 1-a-1, tolerando formato (nome e
  sobrenome concatenados, colunas numéricas omitidas), mas rejeitando IRIs no lugar de
  nomes ou valores diferentes.

Resultado com `gemma4:12b-nvfp4`: **9/16 com resultado idêntico** ao gabarito e **10/16
com conteúdo correto**. Nenhuma query gerada foi rejeitada por sintaxe. As perguntas
resolvidas na primeira tentativa levam de 5 a 36 s cada (Apple M4 16 GB, geração mais
execução do SPARQL; a 05 sozinha leva ~27 s no Ontop).

Os casos que não bateram exatamente, e o que o modelo errou em cada um:

| Query | Erro |
|---|---|
| 02 | contou por `:hasDirector` em vez de `?x a :Director` |
| 08 | subconsulta agregada usando uma variável do padrão externo, que o escopo do SPARQL não propaga |
| 09 | omitiu a coluna `?year` (só formato: o conteúdo bate) |
| 12, 13 | literal de nome fora da forma da base: `"Samuel"` em vez de `"Samuel L."`, `" Thurman"` com espaço |
| 14 | `ORDER BY ?year` com `?year` fora do `SELECT DISTINCT` — rejeitado pelo endpoint |
| 15 | ordenou por `COUNT` decrescente sem isolar o máximo |

O padrão é claro: a parte de vocabulário (que classe, que propriedade, que direção da
inversa) sai certa, porque vem da ontologia no prompt. O que escapa é o recorte do
resultado — quais colunas devolver, como expressar "o que mais…" — e a forma exata dos
literais, que nenhuma das duas fontes do prompt descreve.

**Limitações:** a avaliação é pequena (16 perguntas, várias estruturalmente parecidas com
os exemplos) — é um teste de fumaça, não um benchmark. Títulos estão no original,
geralmente em inglês e com artigo no fim ("Godfather, The"), então perguntas com títulos
traduzidos tendem a não encontrar o filme.

## Licença

- Ontologia, mapeamentos, queries e documentação: [CC0 1.0](LICENSE-CC0) (domínio público).
- Código: [MIT-0](LICENSE).

As licenças cobrem apenas o que foi criado neste repositório. **Os dados não fazem parte
dele:** são baixados do servidor da CTU (origem: kt.ijs.si), derivados do IMDb e sujeitos
aos termos de uso próprios. Verifique-os antes de redistribuir dumps ou triplas
materializadas.
