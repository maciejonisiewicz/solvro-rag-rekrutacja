# Asystent Przeglądu Literatury RAG (OpenReview 2024)

System Retrieval-Augmented Generation (RAG) wspomagający przegląd literatury naukowej z zakresu Machine Learningu na podstawie publikacji i recenzji eksperckich pobranych przez **OpenReview API v2**. Rozwiązanie zestawia deklaracje autorów (abstrakty) z krytyczną oceną recenzentów (zalety, wady, ograniczenia eksperymentalne) oraz zachowuje pełną identyfikowalność źródeł.

> **Wariant realizacji (zgodnie z wytycznymi zadania):** Projekt realizuje pełny potok przetwarzania i wyszukiwania danych (pobieranie z API, lokalna baza wektorowa i słownikowa, wyszukiwanie hybrydowe RRF, pętla decyzyjna Agenta z narzędziami, detekcja braku danych oraz budowanie ustrukturyzowanego promptu końcowego ze źródłami i interfejsem w Streamlit), **bez podłączonego komercyjnego modelu generatywnego LLM**. Zgodnie z instrukcją zadania brak generacji został jawnie oznaczony, a ewaluacja nie opiera się na ręcznie pisanych odpowiedziach.

---

## 1. Pobranie danych, dobór próby i różnice w schematach (`fetch_data.py`)

Dane zostały pobrane z dwóch konferencji korzystających z OpenReview API v2:
- **ICLR 2024** (`ICLR.cc/2024/Conference`)
- **NeurIPS 2024** (`NeurIPS.cc/2024/Conference`)

### Metodologia doboru próby
1. Z każdej konferencji pobrano pulę pierwszych 200 zaakceptowanych publikacji (`content.venueid = venue_id`) wraz z pełnymi wątkami dyskusji (`details='replies'`).
2. Wyselekcjonowano po **40 publikacji z każdej konferencji** (**łącznie 80 prac oraz 308 oficjalnych recenzji**), które posiadały publiczne recenzje oraz dotyczyły wspólnej tematyki **modeli językowych (LLM) i optymalizacji kosztu inferencji** (filtrowanie po słowach kluczowych w tytule, abstrakcie i polu `keywords`, m.in.: *language model, llm, inference, quantization, pruning, attention, kv cache, speculative, decoding, reasoning*).
3. Pełne metadane procesu (dokładny znacznik czasu UTC pobrania, użyte słowa kluczowe, lista identyfikatorów `paper_id` oraz statystyki odrzuconych notatek) są zapisane w pliku `data/sampling_metadata.json`.

### Filtrowanie wątku i różnice między konferencjami
- **Odrzucanie szumu z wątków (`invitations`):** Wątki na OpenReview zawierają wiele typów notatek. Skrypt filtruje wyłącznie oficjalne recenzje (`/-/Official_Review`), odrzucając komentarze, odpowiedzi autorów i decyzje:
  - W próbie **ICLR 2024** wyodrębniono **153 oficjalne recenzje**, odrzucając 682 inne notatki (`Official_Comment`: 488, `Edit`: 111, `Decision`: 40, `Meta_Review`: 40, `Public_Comment`: 3).
  - W próbie **NeurIPS 2024** wyodrębniono **155 oficjalnych recenzji**, odrzucając 1067 innych notatek (`Edit`: 533, `Official_Comment`: 304, `Rebuttal`: 154, `Decision`: 40, `Author_Rebuttal`: 35, `Public_Comment`: 1).
- **Różnice w schematach recenzji:**
  - **Pola tekstowe:** NeurIPS 2024 posiada dedykowane pole `limitations` (obok `summary`, `strengths`, `weaknesses`, `questions`). W ICLR 2024 osobne pole `limitations` nie występuje — uwagi o ograniczeniach recenzenci umieszczają wewnątrz sekcji `weaknesses`.
  - **Skale i formaty ocen (`rating`):** W NeurIPS 2024 oceny zapisywane są jako wartości liczbowe (np. `7`), natomiast w ICLR 2024 jako ciągi znaków łączące ocenę z opisem słownym (np. `"8: accept, good paper"`, `"5: marginally below the acceptance threshold"`). System zachowuje oryginalne nazwy pól i formaty w `data/reviews.json`.

---

## 2. Przechowywanie danych i architektura Retrieval (`rag_system.py`)

Zastosowano podejście wielosilnikowe oparte na lokalnej bazie wektorowej (**ChromaDB**), indeksie słownikowym (**BM25**) oraz fuzji hybrydowej (**RRF**):
- **Jednostka indeksowania i relacja publikacja–recenzja:** Jeden dokument w indeksie odpowiada dokładnie jednej oficjalnej recenzji (`review_id`), powiązanej z publikacją przez `paper_id` w metadanych walidowanych przez modele **Pydantic**. Pozwala to na filtrowanie po konferencji (`where={"conference": ...}`) oraz natychmiastowe zestawienie abstraktu autorów z uwagami konkretnego recenzenta.
- **Wyszukiwanie wektorowe:** Lokalna baza `ChromaDB` (`PersistentClient` z indeksem HNSW i metryką podobieństwa cosinusowego `cosine`) oraz model osadzeń `sentence-transformers/all-MiniLM-L6-v2` (384 wymiary).
- **Punkt odniesienia (Baseline):** Algorytm `BM25Okapi` (`rank_bm25`).
- **Wyszukiwanie hybrydowe (Reciprocal Rank Fusion — RRF):** Łączy ranking wektorowy i leksykalny według wzoru $\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{vec}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$.

### Opisany eksperyment inżynieryjny: Problem ucinania kontekstu (Truncation)
W pierwszej iteracji systemu dokument przekazywany do modelu embeddingów był składany w kolejności: `Tytuł + Abstrakt + Treść recenzji`. Ponieważ model `all-MiniLM-L6-v2` posiada limit wejściowy 256 tokenów, długie abstrakty (ok. 200–250 słów) całkowicie wypełniały okno modelu, a właściwa treść recenzji była ucinana przed wygenerowaniem wektora. W efekcie wszystkie recenzje jednej pracy otrzymywały niemal identyczny wektor, a metryka `Hit@5` wynosiła zaledwie **30.0%**.  
**Rozwiązanie:** Przebudowano strukturę dokumentu kodowanego wektorowo tak, aby na początku znajdowały się tytuł, słowa kluczowe oraz kluczowe sekcje recenzji (`Weaknesses`, `Limitations`, `Strengths`, `Summary`), natomiast pełny abstrakt autorów przeniesiono do metadanych dokumentu. Zmiana ta podniosła wynik `Hit@5` bazy wektorowej z **30.0% do 90.0%** (na poziomie konkretnej recenzji) oraz **100.0%** (na poziomie publikacji).

---

## 3. Agent z narzędziami, budowanie promptu i interfejs (`agent_rag.py`, `app.py`)

Moduł `agent_rag.py` implementuje wieloetapowego Agenta z twardym limitem kroków (`max_steps = 3`) oraz pełnym logowaniem przebiegu rozumowania (`data/agent_trace_log.json`):
1. **Krok 1 (`vector_search`):** Wykonuje wyszukiwanie semantyczne i weryfikuje próg podobieństwa (`similarity_threshold = 0.38`). Jeśli najwyższy wynik spada poniżej progu, agent natychmiast przerywa pracę i zgłasza brak danych w bazie.
2. **Krok 2 (`refine_and_hybrid_search`):** Jeśli pewność semantyczna jest umiarkowana (`< 0.55`) lub zapytanie zawiera specyficzne terminy techniczne/liczby (np. `Rouge-L`, `Equation 3`, `PaLM 8B`), agent automatycznie ekstrahuje słowa kluczowe i uruchamia wyszukiwanie hybrydowe RRF.
3. **Krok 3 (`fetch_additional_paper_reviews`):** Dla najwyżej ocenionej publikacji agent dociąga z bazy dodatkową recenzję tego samego artykułu, aby skonfrontować opinie niezależnych recenzentów w końcowym kontekście.

- Każde zapytanie zapisuje pełny kontekst oraz zbudowany prompt w pliku `data/last_query_context.json`.
- Plik `app.py` udostępnia interfejs webowy w **Streamlit** z podglądem kroków Agenta, wyborem trybu wyszukiwania, filtrami konferencji, kartami porównującymi deklaracje autorów z uwagami recenzentów oraz interaktywnym panelem ewaluacji.

---

## 4. Ewaluacja jakości i rozszerzenia ("Jeśli chcesz pójść dalej" — `evaluate.py`)

Przygotowano zestaw ewaluacyjny `data/eval_questions.json` składający się z **12 pytań**:
- **10 pytań z odpowiedzią w bazie** — z ręcznie przypisanym identyfikatorem docelowej pracy (`target_paper_id`), docelowej recenzji (`target_review_id`) oraz pisemnym uzasadnieniem trafności.
- **2 pytania bez wystarczających danych w bazie** (`Q11` – pytanie publicystyczne o bezużyteczność AI; `Q12` – pytanie medyczne o rekonstrukcję więzadeł kolana).

### Zrealizowane kierunki rozszerzeń ("Jeśli chcesz pójść dalej")
1. **Agent korzystający z narzędzi (`max_steps = 3`):** Poprawia wynik względem pojedynczego wyszukania wektorowego z **90.0% do 100.0% Hit@5** oraz podnosi `MRR` z **0.531 do 0.708** (`nDCG@5` z **0.674 do 0.851**).
2. **Wyszukiwanie hybrydowe (RRF: ChromaDB + BM25):** Łączy rozumienie semantyczne z dopasowaniem rzadkich terminów, osiągając **100.0% Hit@5** oraz najwyższe `nDCG@5` (**0.875**).
3. **Dodatkowe metryki (`MRR` oraz `nDCG@5`) i ich uzasadnienie ponad `Hit@5`:**
   - `Hit@5` jest metryką binarną (traktuje identycznie źródło na 1. i 5. pozycji).
   - `MRR (Mean Reciprocal Rank)` mierzy odwrotność pozycji pierwszego trafnego wyniku ($\frac{1}{\text{rank}}$), premiując umieszczenie właściwej recenzji na samym szczycie kontekstu.
   - `nDCG@5 (Normalized Discounted Cumulative Gain)` wprowadza stopniowaną trafność ($rel=2$ dla docelowej recenzji omawiającej dany zarzut, $rel=1$ dla innej recenzji właściwej publikacji, $rel=0$ dla pozostałych prac), oceniając jakość całego pakietu 5 dokumentów przekazywanych do promptu.
4. **Badanie ablacyjne: Dołączenie recenzji vs korzystanie wyłącznie z abstraktów (`search_abstracts_only`):** Podczas gdy wyszukiwanie uwzględniające recenzje osiąga **100.0% Hit@5** (`MRR = 0.808` dla BM25 oraz `0.800` dla Hybrydy RRF), ograniczenie indeksu wyłącznie do abstraktów (po deduplikacji publikacji) powoduje spadek `Hit@5` na poziomie docelowej recenzji do **20.0%**, `MRR` do **0.200** oraz `nDCG@5` do **0.285**. Wynika to z faktu, że abstrakty autorów nie zawierają informacji o szczegółowych zarzutach metodologicznych (np. krytyce metryki w konkretnym równaniu czy brakujących baseline'ach) ani nie pozwalają zidentyfikować, który z recenzentów wskazał dane ograniczenie.
5. **Analiza wpływu doboru próby i dostępności recenzji na różnorodność polecanych publikacji:**
   - Na 10 zapytań testowych w Top-5 system poleca **33 unikalne publikacje z 80 dostępnych w bazie** (**41.2% pokrycia katalogu — Catalog Coverage**), co świadczy o braku dominacji pojedynczych artykułów (brak efektu *hubness*).
   - Średnia liczba unikalnych publikacji w Top-5 na jedno zapytanie wynosi **3.50 / 5**, co pozwala jednocześnie zachować różnorodność prac i dostarczyć więcej niż jedną recenzję dla najbardziej trafnego artykułu.
   - Rozkład polecanych prac między konferencjami jest zbalansowany (**46.0% ICLR 2024** vs **54.0% NeurIPS 2024**).

### Tabela porównawcza wszystkich metod
| Metoda / Wariant Retrieval | Hit@5 (Publikacje — deduplikowane) | Hit@5 (Konkretna Recenzja) | MRR (Poziom Recenzji) | nDCG@5 (Stopniowana trafność) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Wyszukiwanie Wektorowe (ChromaDB)** | 100.0% (10/10) | 90.0% (9/10) | 0.531 | 0.674 |
| **2. Baseline (BM25 — Abstrakty + Recenzje)** | 100.0% (10/10) | 100.0% (10/10) | **0.808** | 0.869 |
| **3. Wyszukiwanie Hybrydowe (RRF: Wektor + BM25)** | **100.0%** (10/10) | **100.0%** (10/10) | 0.800 | **0.875** |
| **4. Agent Wieloetapowy z Narzędziami (Limit = 3 kroki)** | **100.0%** (10/10) | **100.0%** (10/10) | 0.708 | 0.851 |
| **5. Tylko Abstrakty (Bez recenzji — Ablacja)** | 90.0% (9/10) | 20.0% (2/10) | 0.200 | 0.285 |

- **Zachowanie na pytaniach bez danych (`Q11`, `Q12`):** Oba zapytania spoza domeny uzyskały podobieństwo cosinusowe poniżej progu `0.38` (`0.292` dla `Q11` oraz `0.168` dla `Q12`) i zostały w **100.0% (2/2) poprawnie odrzucone**.

### Analiza przypadku nietrafnego i jego naprawy (Error Analysis — `Q04`)
- **Zapytanie `Q04`:** *"Which review criticizes the use of Rouge-L in Equation 3 as too weak for free-generation results and suggests replacing it with an entailment model?"* (Oczekiwane źródło: praca `AqN23oqraW`, recenzja `2mTTJZa647`).
- **Wynik pojedynczego wyszukania wektorowego:** Pozycja **#7 (`MISS` dla Hit@5)** — na 1. miejscu zwrócono pracę *Controlled Text Generation via Language Model Arithmetic* (`score: 0.479`).
- **Przyczyna i naprawa:** Zapytanie nie zawiera nazwy własnej benchmarku (*KoLA*), lecz odwołuje się do szczegółowego zarzutu wewnątrz sekcji `weaknesses` (`Rouge-L` w `Equation 3`). Model gęstych osadzeń (`all-MiniLM-L6-v2`) kompresuje całą recenzję do pojedynczego wektora 384D, przez co nazwa pojedynczej metryki ginie w ogólnym temacie generacji tekstu. Zastosowanie **Wyszukiwania Hybrydowego RRF** oraz **Agenta z narzędziami** (który w 2. kroku wyodrębnia terminy techniczne i uruchamia fuzję z BM25) naprawia ten błąd i przesuwa docelową recenzję na **1. miejsce (`#1`)**.

---

## 5. Struktura projektu i instrukcja uruchomienia

```text
├── data/
│   ├── papers.json                # 80 pobranych publikacji (ICLR 2024 & NeurIPS 2024)
│   ├── reviews.json               # 308 oficjalnych recenzji z ocenami i polami
│   ├── sampling_metadata.json     # Metadane doboru próby, identyfikatory i data pobrania
│   ├── eval_questions.json        # 12 pytań ewaluacyjnych z ground-truth i uzasadnieniem
│   ├── eval_results.json          # Pełne wyniki eksperymentów (Hit@5, MRR, nDCG@5, Różnorodność)
│   ├── last_query_context.json    # Zapisany kontekst i prompt z ostatniego zapytania
│   └── agent_trace_log.json       # Zapisany przebieg kroków Agenta z narzędziami
├── fetch_data.py                  # Pobieranie i filtrowanie danych z OpenReview API v2
├── rag_system.py                  # Silnik RAG (ChromaDB, BM25, Hybryda RRF, Ablacja abstraktów)
├── agent_rag.py                   # Agent z narzędziami (limit 3 kroków), budowanie promptu i CLI
├── evaluate.py                    # Kompleksowa ewaluacja wszystkich 5 kierunków rozszerzeń
├── app.py                         # Aplikacja webowa w Streamlit
└── requirements.txt               # Zależności projektu
```

### Instrukcja uruchomienia

1. Utworzenie środowiska i instalacja zależności:
```bash
python3 -m venv .venv
source .venv\Scripts\activate. #na macu .venv/bin/activate
pip install -r requirements.txt
```

2. (Opcjonalnie) Pobranie danych z OpenReview API v2:
```bash
python fetch_data.py
```
*Pobrane dane (80 publikacji i 308 recenzji) są już zapisane lokalnie w katalogu `data/`.*

3. Uruchomienie skryptu ewaluacyjnego w konsoli:
```bash
python evaluate.py
```

4. Uruchomienie aplikacji webowej (Streamlit):
```bash
streamlit run app.py
```