# Sugestões de perguntas derivadas da base de conhecimento — Design

**Data:** 2026-08-12
**Status:** Aprovado em conversa
**Contexto:** As pills de "experimente perguntar" no chat são três strings fixas em `translations.py` (`example_q1..q3`), escritas para os papers padrão (attention, RAG, BERT). Quem substitui a base — sobe outros PDFs, remove documentos ou limpa tudo — continua vendo sugestões sobre BERT, que a base não responde mais. As sugestões passam a ser derivadas do conteúdo indexado.

## Decisões

| Decisão | Escolha |
|---|---|
| Geração | LLM (Groq) a partir dos trechos indexados, com fallback determinístico por título |
| Momento | Lazy: geradas quando o chat pede, com cache por impressão digital da base |
| Camada | Núcleo (`rag/`) + endpoint novo na FastAPI; a UI nunca fala com o LLM |
| Modelo | `llama-3.1-8b-instant` (novo `SUGGESTION_MODEL`) — sugestão não precisa do 70B |
| Quantidade | 3 perguntas, como hoje |
| Idioma | en/pt, gerado no idioma pedido; cache por `(impressão digital, idioma)` |

## Componentes

### `src/rag/generation/suggestions.py` (novo)

```python
def suggest_questions(chat: GroqChat, chunks: list[Chunk],
                      language: str = "en", n: int = 3) -> list[str]
```

Segue o formato de `rewriter.py`: `_SYSTEMS` por idioma, uma chamada ao `chat.complete`, e degradação silenciosa em vez de exceção.

- **Amostra para o prompt:** o primeiro chunk de cada documento (até 5 documentos, na ordem de `doc_id`), truncado em ~600 caracteres. O primeiro chunk costuma cobrir abstract/introdução, que é onde está o assunto do paper; a amostra fixa mantém o prompt pequeno e o resultado estável entre chamadas da mesma base.
- **System prompt:** pede exatamente `n` perguntas curtas, respondíveis pelos trechos, uma por linha, sem numeração nem comentários. Versões en/pt.
- **Parser:** quebra por linha, remove numeração (`1.`, `1)`), bullets (`-`, `*`) e aspas, descarta linhas vazias, corta em `n`.
- **Fallback determinístico:** dispara quando `chat.complete` levanta `GenerationError` ou quando o parser devolve menos de `n` linhas válidas. Monta as perguntas a partir dos títulos com templates por idioma, ciclando os títulos até fechar `n`:
  - en: `What does {title} propose?`, `What are the main findings in {title}?`, `How does {title} evaluate its approach?`
  - pt: `O que o {title} propõe?`, `Quais são os principais resultados de {title}?`, `Como {title} avalia a abordagem?`
- **Base vazia:** `chunks == []` → retorna `[]` sem chamar o LLM.

### `src/rag/config.py`

Acrescenta `SUGGESTION_MODEL = "llama-3.1-8b-instant"`.

### `src/rag/service.py`

```python
def suggested_questions(self, language: str = "en") -> list[str]
```

Calcula a impressão digital da base (md5 sobre os pares `(doc_id, nº de chunks)` ordenados) e memoiza em `self._suggestions_cache`, um par `(impressão digital, {idioma: perguntas})`. Quando a impressão digital muda, o dicionário de idiomas é descartado inteiro — o cache guarda só a base corrente e não cresce a cada alteração. `add_document`, `remove_document`, `reset_documents` e `restore_default_documents` mudam o conjunto de `doc_id` ou a contagem de chunks, então a chave muda sozinha — nenhum dos quatro caminhos de escrita precisa invalidar cache explicitamente. O cache mora no serviço, que é singleton nos dois modos, e portanto vale para HTTP e embedded com um código só.

### `src/api/main.py` e `src/api/schemas.py`

`GET /suggestions?language=en` → `SuggestionsResponse(questions: list[str])`. Fora do rate limit: é leitura e o resultado vem do cache do serviço.

### `src/app/api_client.py` e `src/app/backend.py`

`suggestions(language: str = "en") -> list[str]` — via HTTP no `api_client`, direto no serviço no ramo embedded do `backend.py`, exportado nos dois ramos do `if _mode()`, mesmo contrato de `documents()`.

### `src/app/views/chat.py`

O bloco de pills troca as três strings fixas por `suggestions(lang)`. Lista vazia ou `ApiError` → as pills não são renderizadas, sem mensagem de erro: são decoração e não devem derrubar a página nem poluir a tela quando o backend está fora. O label `try_asking` continua.

### `src/app/translations.py`

Remove as chaves `example_q1`, `example_q2` e `example_q3` (en e pt). O fallback passou para o núcleo, que é quem conhece os títulos dos documentos.

## Testes

- `tests/test_suggestions.py` (novo): parsing remove numeração, bullets e aspas e corta em `n`; `GenerationError` cai no fallback por títulos; base vazia devolve `[]`; o fallback fecha `n` perguntas mesmo com um único documento; a amostra do prompt limita-se a 5 documentos.
- `tests/test_service.py`: a segunda chamada de `suggested_questions` não consome resposta do `FakeGroq` (cache ativo) e volta a consumir depois de `remove_document` (invalidação pela impressão digital).
- `tests/test_api.py`: shape de `GET /suggestions`.
- `tests/test_backend_embedded.py`: `suggestions()` no modo embedded devolve lista.

## Fora de escopo

- Persistir as sugestões em disco junto do índice (o cache é por processo; um restart regenera)
- Sugestões personalizadas por histórico do visitante
- Regenerar sugestões durante o upload (permanece lazy)

## Critérios de sucesso

1. Trocar a base (upload de outros PDFs, remoção ou reset + restore) muda as pills do chat no próximo carregamento
2. Base inalterada não gera chamada nova ao Groq
3. Sem `GROQ_API_KEY` ou com o Groq fora, as pills ainda aparecem via fallback por títulos
4. Base vazia continua mostrando o card de "base vazia", sem pills
5. Seletor EN/PT devolve sugestões no idioma escolhido
6. Suíte offline verde nos dois modos (HTTP e embedded)
