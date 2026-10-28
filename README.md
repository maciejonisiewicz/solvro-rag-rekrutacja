# Asystent przeglądu literatury RAG — OpenReview 2024

Aplikacja do przeszukiwania publikacji naukowych i oficjalnych recenzji z konferencji **ICLR 2024** oraz **NeurIPS 2024**. Łączy abstrakty autorów z ocenami recenzentów, prezentuje źródła i umożliwia porównanie metod wyszukiwania w interfejsie Streamlit.

> **Zakres:** jest to system retrieval i przygotowywania kontekstu dla RAG, **bez podłączonego generatywnego LLM**. Agent jest deterministycznym kontrolerem narzędzi wyszukiwawczych, a nie autonomicznym modelem językowym. Wyniki ewaluacji dotyczą odnajdywania źródeł, nie jakości wygenerowanych odpowiedzi.

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

## Instalacja i uruchomienie

Wymagany Python i zależności z `requirements.txt`. W katalogu projektu:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.fileWatcherType none
```

Na Windows aktywacja środowiska wygląda inaczej (`.venv\Scripts\activate`). Pierwsze uruchomienie może wymagać pobrania modelu embeddingowego. Dane i indeks lokalny muszą być dostępne w katalogu `data/`.

Aby ponownie wykonać ewaluację, użyj zakładki **Ewaluacja** w aplikacji lub uruchom:

```bash
python evaluate.py
```

Zbiór pytań referencyjnych znajduje się w `data/eval_questions.json`, a zapis wyników w `data/eval_results.json`. Uruchomienie ewaluacji nie powinno nadpisywać ręcznie oznaczonego ground truth.

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
app.py                 # interfejs Streamlit
agent_rag.py           # deterministyczny agent i kontekst źródłowy
rag_system.py          # wyszukiwanie wektorowe, BM25 i hybrydowe
evaluate.py            # metryki i raport ewaluacji
fetch_data.py          # pobieranie danych z OpenReview
requirements.txt       # zależności
README.md              # dokumentacja
data/
  papers.json
  reviews.json
  sampling_metadata.json
  eval_questions.json
  eval_results.json
  chroma_db/
```

## Ograniczenia i możliwe rozwinięcia

- Niewielki zbiór ewaluacyjny (8 pytań z docelową recenzją) ogranicza możliwość uogólniania wyników.
- Dodatkowe sortowanie agenta na podstawie terminów technicznych jest heurystyką, a nie trenowanym rerankerem; wymaga testów na większym zbiorze.
- Próg odrzucenia 0.38 nie odrzucił jednego z dwóch pytań bez danych w dodatkowym teście holdout; wymaga odrębnego strojenia i walidacji.
- Metryki retrieval nie oceniają prawdziwości ani jakości końcowej odpowiedzi LLM, ponieważ model generatywny nie jest podłączony.
- Dodatkowy holdout jest mały (4 pytania z odpowiedzią, 2 bez danych), więc jego wyniki również nie pozwalają wnioskować o skuteczności na szerokiej populacji zapytań.
- Możliwe rozwinięcia: większy niezależnie przygotowany zbiór testowy, analiza czasu odpowiedzi i kosztu, reranker cross-encoder oraz generacja odpowiedzi z cytowaniami.

## Powtarzalność

Wyniki pochodzą z raportów `data/eval_results.json` (zbiór główny, ewaluacja v3) oraz `data/eval_holdout_results.json` (dodatkowy test). Pytania i ręczne oznaczenia przechowywane są oddzielnie od indeksowanych źródeł. Przy powtarzaniu porównań należy zachować wersję danych, konfigurację indeksu, próg odrzucenia i ten sam zestaw pytań. Testy metryk: `python -m unittest -v test_evaluate`; test indeksu: `python -m unittest -v test_rag_index`.
