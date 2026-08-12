# Base de conhecimento por sessão no demo hospedado — Design

**Data:** 2026-08-12
**Status:** Aprovado em conversa
**Contexto:** O demo hospedado tem uma base só, compartilhada por todos os visitantes. O `RAGService` é construído uma vez por processo (`st.cache_resource` em `backend.py`) e todo mundo usa o mesmo `IndexStore`, que grava em `data/index` no disco do container. Duas pessoas testando assuntos diferentes ao mesmo tempo se atrapalham, e "Limpar tudo" apaga os documentos de todos os visitantes ativos. Cada sessão passa a ter sua própria base.

## Decisões

| Decisão | Escolha |
|---|---|
| Isolamento | Por sessão do Streamlit, sem login |
| Estado inicial | Cópia da coleção padrão (5 papers do `data/index`), para manter o "zero setup para o visitante" |
| Persistência | Nenhuma: a base vale enquanto a aba estiver aberta; um F5 recomeça da coleção padrão, com aviso na UI |
| Índice em disco | Passa a ser somente leitura — é o molde de onde toda sessão nasce |
| Rate limit | Continua global, de propósito: protege a cota única do Groq, não o visitante |
| Banco de feedback | Continua compartilhado — as métricas são agregadas por natureza |
| Modo HTTP local | Inalterado: um usuário só, persistindo em disco como hoje |

## Componentes

### `src/rag/service.py`

`RAGService.__init__` ganha `persist: bool = True`.

Com `persist=False`:
- `add_document` indexa o PDF e **não** chama `store.save(self.index_dir)`; apaga o arquivo temporário logo após a indexação (`ingest_pdf` precisa de um caminho em disco, mas depois de indexado o PDF não tem mais uso — os chunks estão em memória)
- `remove_document` e `reset_documents` mexem só no `store` em memória, sem `save`

Isso é o ponto crítico do desenho: hoje toda escrita chama `store.save(self.index_dir)`. Mantido, a sessão de um visitante sobrescreveria o índice padrão no disco — trocaria "atrapalha os outros agora" por "corrompe o molde para sempre".

`restore_default_documents` não muda: continua baixando do arXiv, e continua sendo o caminho do modo HTTP local. O modo embutido deixa de chamá-lo (ver abaixo).

### `src/app/backend.py` (ramo embutido)

A construção se parte em dois.

**Compartilhado**, via `st.cache_resource` chaveado pela versão do código (mecanismo atual do `_ensure_fresh_rag`, intacto): `Embedder`, `Reranker`, `GroqChat` e `FeedbackDB`. São caros de carregar e não têm estado por visitante.

**Por sessão**, em `st.session_state`: um `IndexStore` novo via `IndexStore.load(INDEX_DIR)` (~167 chunks, ~1-2 MB, ~200 ms) e um `RAGService` construído com `persist=False`, com os recursos compartilhados injetados e um `documents_dir` temporário exclusivo da sessão (`tempfile.mkdtemp`). O diretório por sessão evita a corrida de dois visitantes subindo arquivos de mesmo nome ao mesmo tempo; como o PDF é apagado após a indexação, ele fica vazio.

`_build_service()` mantém o nome e passa a devolver o serviço da sessão — os testes existentes que fazem monkeypatch nele continuam valendo.

O acesso ao `st.session_state` fica atrás de um helper de uma linha, `_session_cache() -> dict`. Fora de um runtime do Streamlit o `st.session_state` não é confiável, então é esse helper que os testes trocam por um dicionário comum — é assim que "duas sessões" viram dois dicionários e o isolamento fica testável offline.

**A armadilha do Streamlit Cloud reaparece num lugar novo.** `_ensure_fresh_rag()` purga o pacote `rag` de `sys.modules` e chama `st.cache_resource.clear()` quando os fontes mudam, mas um `RAGService` guardado em `st.session_state` sobrevive a isso e continua segurando classes da geração anterior — o mesmo `AttributeError` que os commits `b70a214`/`611c0ab` resolveram, agora escondido na sessão. Então o serviço da sessão é guardado como `(versão, serviço)` e recriado quando a versão muda.

`restore_defaults()` no modo embutido passa a descartar a entrada do `session_state` e deixar a próxima chamada recriar o serviço a partir do índice em disco: instantâneo, sem rede, sem depender do arXiv estar de pé. Devolve o mesmo shape de hoje (`documents_added`, `chunks_added`), contados a partir do store recriado.

### `src/app/views/documents.py` e `src/app/translations.py`

Um aviso discreto na página Documentos: a base é desta aba e vale enquanto ela estiver aberta. Texto en/pt em `translations.py`. Sem banner de alerta — é informação, não erro.

## Testes

- `tests/test_service.py`: com `persist=False`, `add_document`, `remove_document` e `reset_documents` deixam `index_dir` intocado (nenhum arquivo criado ou modificado) e o PDF temporário some após a indexação; com `persist=True` o comportamento atual segue idêntico.
- `tests/test_backend_embedded.py`: duas sessões simuladas enxergam bases diferentes — uma indexa um documento e a outra continua só com a coleção padrão; `restore_defaults` recria a sessão a partir do disco sem chamar o download do arXiv; o serviço da sessão é recriado quando a versão do código muda.
- Suíte do modo HTTP intacta (`persist=True` é o default).

## Fora de escopo

- Trava de tamanho ou quantidade de upload por sessão. Este desenho isola visitantes entre si; ele **não** protege contra um visitante mal-intencionado, que ainda pode consumir a memória do container com um PDF gigante e derrubar o app para todos. Trabalho separado.
- Sobreviver ao refresh da página (ID no navegador com expiração no servidor)
- Contas, login e bases persistentes entre visitas
- Isolamento do banco de feedback

## Critérios de sucesso

1. Duas abas simultâneas: subir um PDF numa não muda a lista de documentos nem as respostas da outra
2. "Limpar tudo" numa aba não afeta a outra
3. Recarregar a página devolve a coleção padrão dos 5 papers, e a UI avisa que é assim
4. `data/index` no disco não é modificado por nenhuma ação do visitante
5. "Restaurar coleção padrão" no demo responde na hora, sem baixar nada
6. Comportamento local (modo HTTP) inalterado; suíte offline verde
