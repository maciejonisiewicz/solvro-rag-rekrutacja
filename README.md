# Asystent przeglądu literatury RAG — OpenReview 2024

Aplikacja do przeszukiwania publikacji naukowych i oficjalnych recenzji z konferencji **ICLR 2024** oraz **NeurIPS 2024**. Łączy abstrakty autorów z ocenami recenzentów, prezentuje źródła i umożliwia porównanie metod wyszukiwania w interfejsie Streamlit.

> **Zakres:** jest to system retrieval i przygotowywania kontekstu dla RAG, **bez podłączonego generatywnego LLM**. Agent jest deterministycznym kontrolerem narzędzi wyszukiwawczych, a nie autonomicznym modelem językowym. Wyniki ewaluacji dotyczą odnajdywania źródeł, nie jakości wygenerowanych odpowiedzi.

## Podgląd aplikacji

![Interfejs asystenta przeglądu literatury RAG — wyniki wyszukiwania recenzji OpenReview](assets/app-preview.png)

*Przykład wyszukiwania recenzji publikacji KoLA. Interfejs pokazuje źródła, deklaracje autorów i uwagi recenzenta.*
  
## Dane

- **80 publikacji**: po 40 z ICLR 2024 i NeurIPS 2024.
- **308 oficjalnych recenzji** powiązanych z publikacjami.
- Dane: `data/papers.json`, `data/reviews.json`; metadane doboru próby: `data/sampling_metadata.json`.
- Źródło: OpenReview API v2. Uwzględniane są oficjalne recenzje, a nie komentarze, decyzje i odpowiedzi autorów.

## Metody wyszukiwania

1. **Wektorowe:** podobieństwo semantyczne, lokalny indeks ChromaDB.
2. **BM25:** dopasowanie leksykalne do tekstu abstraktów i recenzji.
3. **Hybrydowe RRF:** łączenie rankingów wektorowego i BM25.
4. **Agent deterministyczny:** maksymalnie trzy kroki narzędziowe; rozpoczyna od wyszukiwania wektorowego, w razie potrzeby przeformułowuje zapytanie i uruchamia hybrydę, stosuje dodatkowe uporządkowanie według terminów technicznych i może pobrać uzupełniający kontekst. Kontekst dodatkowy nie jest traktowany jako wynik rankingu.
5. **Tylko abstrakty:** wariant ablacyjny bez treści recenzji.

Aplikacja pokazuje tytuły prac, identyfikatory i odnośniki OpenReview, fragmenty abstraktów i ocen recenzentów, przebieg narzędzi agenta oraz kontekst do ewentualnej generacji odpowiedzi.



## Wiarygodność kontekstu i ablacja

Wyszukiwanie recenzji wykorzystuje pola `summary`, `strengths`, `weaknesses`, `limitations` oraz `questions`. W znalezionych źródłach i budowanym promptcie zachowywane są pełne pola recenzji (z ich nazwami), osobny abstrakt autorów oraz identyfikatory i linki OpenReview. Recenzje mogą zawierać opinie i pytania, nie należy traktować ich jako bezspornych faktów.

**Tryb „Tylko abstrakty”** nie przekazuje do interfejsu ani promptu identyfikatorów recenzji, ich ocen i treści. W tym wariancie źródłem jest wyłącznie publikacja. Wariant abstraktowy tworzy oddzielny indeks BM25 z jednym abstraktem na publikację. Zmiana może wpływać na wynik Paper Hit@5, dlatego wyniki trzeba przeliczyć po wdrożeniu.

## Instalacja i uruchomienie

**Wymagania:** Python 3.12 (zalecany), Git i dostęp do internetu przy pierwszym uruchomieniu. Projekt uruchamia się z terminala w VS Code, PyCharmie lub systemowego.

### 1. Pobranie projektu

```bash
git clone https://github.com/maciejonisiewicz/solvro-rag-rekrutacja.git
cd solvro-rag-rekrutacja
```

### 2. Instalacja zależności

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

Sprawdź `python3 --version` (zalecany Python 3.12; na macOS przetestowano także 3.14). Jeśli masz `python3.12`, możesz użyć go zamiast `python3` poniżej.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Uruchomienie aplikacji

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType none
```

**macOS / Linux:**

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

Otwórz `http://localhost:8501`. Indeks ChromaDB zostanie utworzony automatycznie z danych w repozytorium.

### 4. Ewaluacja

Zatrzymaj aplikację (`Ctrl+C`), a następnie uruchom:

**Windows:**

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

**macOS / Linux:**

```bash
python evaluate.py
```

**Sprawdzone konfiguracje:** Windows — Python 3.12 i 3.14; macOS — Python 3.14. Python 3.15 na Windowsie nie przeszedł instalacji zależności.


### Testy automatyczne i dodatkowy holdout

**macOS / Linux:**

```bash
./.venv/bin/python -m unittest -v test_evaluate test_rag_index test_context_integrity
./.venv/bin/python holdout_eval.py --run
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_evaluate test_rag_index test_context_integrity
.\.venv\Scripts\python.exe holdout_eval.py --run
```

Holdout korzysta z zapisanych i wcześniej zweryfikowanych oznaczeń w `data/eval_holdout_verified.json`. Nie służy do dalszego strojenia agenta lub progu odrzucania.

## Metodologia ewaluacji

Zamrożony zbiór główny (`data/eval_questions.json`) zawiera **12 pytań**: 10 z odpowiedzią w danych i 2 bez wystarczających danych. **Paper Hit@5** mierzymy na 10 pytaniach z odpowiedzią, a **Review Hit@5, MRR@5 i nDCG@5** na 8 pytaniach z ręcznie wskazaną konkretną recenzją. Wariant tylko z abstraktami nie ma metryk recenzji (N/A).

**Definicja nDCG@5 (ewaluacja v3):** oceniamy jedną ręcznie wskazaną recenzję jako trafną (`relevance=1`); inne, nieocenione recenzje otrzymują `0`, co *nie oznacza*, że są obiektywnie nietrafne. Przy jednym oznaczonym źródle nDCG@5 i MRR@5 są silnie powiązane. Wyników nDCG v3 nie należy porównywać bezpośrednio z wcześniejszą metodą częściowej trafności recenzji z tej samej publikacji.

### Wyniki głównej ewaluacji (10 pytań z odpowiedzią, 8 ze wskazaną recenzją)

| Metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 (v3) |
|---|---:|---:|---:|---:|
| Wektorowe (ChromaDB) | 100% | 87,5% | 0.542 | 0.624 |
| BM25 (abstrakty + recenzje) | 100% | 100% | 0.917 | 0.938 |
| Hybrydowe (RRF) | 100% | 100% | 0.906 | 0.929 |
| Agent deterministyczny | 100% | 100% | **1.000** | **1.000** |
| Tylko abstrakty | 90% | N/A | N/A | N/A |

Agent umieścił wszystkie osiem oznaczonych recenzji na pierwszym miejscu w *głównym* zbiorze. Heurystyki agenta były jednak dopracowywane przy znajomości tego niewielkiego zestawu, dlatego wyniki mogą być optymistyczne.

### Dodatkowy test holdout (4 nowe pytania z odpowiedzią + 2 bez danych)

Dodatkowe pytania dotyczą innych publikacji niż główne oznaczenia. Zostały ręcznie sprawdzone pod kątem wskazanych źródeł przed uruchomieniem porównania. Test jest **niewielki i eksploracyjny**, a nie statystycznie reprezentatywny. Opis „holdout” oznacza tutaj pytania niewykorzystywane do wcześniejszego dopasowania heurystyk; nie oznacza pełnej niezależności od korpusu i procesu przygotowywania testu.

| Metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 (v3) |
|---|---:|---:|---:|---:|
| Wektorowe (ChromaDB) | 100% | 100% | 0.750 | 0.815 |
| BM25 (abstrakty + recenzje) | 100% | 100% | **0.875** | **0.908** |
| Hybrydowe (RRF) | 100% | 100% | **0.875** | **0.908** |
| Agent deterministyczny | 100% | 100% | 0.750 | 0.815 |
| Tylko abstrakty | 100% | N/A | N/A | N/A |

Na tych czterech nowych pytaniach wszystkie metody odnajdują docelowe recenzje w Top 5, ale BM25 i RRF mają wyższe MRR i nDCG niż agent. **Nie dostrajano agenta na podstawie tych wyników.**

### Analiza błędu Q04

Pytanie z zestawu głównego dotyczy krytyki **ROUGE-L w Equation 3** i propozycji użycia modelu entailment. Oznaczona recenzja: `2mTTJZa647` dla publikacji `AqN23oqraW` (*KoLA: Carefully Benchmarking World Knowledge of Large Language Models*).

- Wektorowe: pozycja **7**.
- BM25: pozycja **1**.
- Hybrydowe: pozycja **1**.
- Agent: pozycja **1**.

To konkretny przykład, w którym dokładne terminy techniczne lepiej wyszukuje metoda leksykalna. Nie dowodzi przewagi BM25 w całej populacji zapytań.

### Odrzucanie pytań bez danych

Przy progu podobieństwa wektorowego **0.38** w zestawie głównym oba pytania bez odpowiedzi zostały odrzucone (Q11: `0.292`, Q12: `0.168`), a wszystkie 10 pytań z odpowiedzią zostało zaakceptowanych. To diagnostyka **progu wektorowego**, a nie osobny pomiar jakości generacji lub całego agenta.

W dodatkowym teście holdout poprawnie odrzucono **1/2** pytań bez danych (H105: `0.140`). Pytanie H106 (`0.384`) zostało błędnie zaakceptowane. Czułość progu jest ograniczeniem systemu; **nie dostrajano progu po obejrzeniu wyniku holdout**.


## Struktura projektu

```text
solvro-rag-rekrutacja/
├── app.py                 # Interfejs Streamlit
├── agent_rag.py           # Agent deterministyczny i kontekst źródłowy
├── rag_system.py          # ChromaDB, BM25 i wyszukiwanie hybrydowe
├── evaluate.py            # Ewaluacja i analiza błędów
├── holdout_eval.py        # Dodatkowy test na nowych pytaniach
├── test_evaluate.py       # Testy metryk
├── test_rag_index.py      # Testy aktualności indeksu
├── test_context_integrity.py  # Testy źródeł i trybu abstraktowego
├── fetch_data.py          # Pobieranie danych z OpenReview
├── requirements.txt       # Zależności Pythona
├── README.md              # Dokumentacja projektu
└── data/
    ├── papers.json
    ├── reviews.json
    ├── sampling_metadata.json
    ├── eval_questions.json
    ├── eval_results.json
    ├── eval_holdout_candidates.json
    ├── eval_holdout_verified.json
    └── eval_holdout_results.json
```

Indeks ChromaDB jest generowany lokalnie przy pierwszym uruchomieniu. Pliki indeksu i logi działania nie są przechowywane w repozytorium.
  

## Ograniczenia i możliwe rozwinięcia

- Niewielki zbiór ewaluacyjny (8 pytań z docelową recenzją) ogranicza możliwość uogólniania wyników.
- Dodatkowe sortowanie agenta na podstawie terminów technicznych jest heurystyką, a nie trenowanym rerankerem; wymaga testów na większym zbiorze.
- Próg odrzucenia 0.38 nie odrzucił jednego z dwóch pytań bez danych w dodatkowym teście holdout; wymaga odrębnego strojenia i walidacji.
- Metryki retrieval nie oceniają prawdziwości ani jakości końcowej odpowiedzi LLM, ponieważ model generatywny nie jest podłączony.
- Dodatkowy holdout jest mały (4 pytania z odpowiedzią, 2 bez danych), więc jego wyniki również nie pozwalają wnioskować o skuteczności na szerokiej populacji zapytań.
- Możliwe rozwinięcia: większy niezależnie przygotowany zbiór testowy, analiza czasu odpowiedzi i kosztu, reranker cross-encoder oraz generacja odpowiedzi z cytowaniami.

## Powtarzalność

Wyniki pochodzą z raportów `data/eval_results.json` (zbiór główny, ewaluacja v3) oraz `data/eval_holdout_results.json` (dodatkowy test). Pytania i ręczne oznaczenia przechowywane są oddzielnie od indeksowanych źródeł. Przy powtarzaniu porównań należy zachować wersję danych, konfigurację indeksu, próg odrzucenia i ten sam zestaw pytań. Testy metryk: `python -m unittest -v test_evaluate`; test indeksu: `python -m unittest -v test_rag_index`.
