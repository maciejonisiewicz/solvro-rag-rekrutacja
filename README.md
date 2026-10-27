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

Zbiór zawiera **12 pytań**: 10 z odpowiedzią w danych i 2 bez wystarczających danych. Metryki **Paper Hit@5** są liczone na 10 pytaniach z odpowiedzią. Metryki **Review Hit@5**, **MRR@5** i **nDCG@5** są liczone wyłącznie na **8 pytaniach z ręcznie oznaczoną docelową recenzją**. Dla wariantu tylko z abstraktami metryki recenzji nie mają zastosowania (N/A).

### Wyniki lokalnej ewaluacji

| Metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Wektorowe (ChromaDB) | 100% | 87,5% | 0.542 | 0.615 |
| BM25 (abstrakty + recenzje) | 100% | 100% | 0.917 | **0.930** |
| Hybrydowe (RRF) | 100% | 100% | 0.906 | 0.870 |
| Agent deterministyczny | 100% | 100% | **1.000** | 0.903 |
| Tylko abstrakty | 90% | N/A | N/A | N/A |

**Interpretacja:** agent osiąga najwyższe MRR@5, natomiast BM25 osiąga najwyższe nDCG@5. Samo dodanie hybrydy nie poprawia każdej metryki względem BM25.

### Analiza błędu: Q04

Pytanie dotyczy krytyki **ROUGE-L w Equation 3** i propozycji użycia modelu entailment. Prawidłowa recenzja: `2mTTJZa647` dla publikacji `AqN23oqraW` (*KoLA: Carefully Benchmarking World Knowledge of Large Language Models*).

- Wektorowe: pozycja **7**.
- BM25: pozycja **1**.
- Hybrydowe: pozycja **1**.
- Agent po poprawkach: pozycja **1**.

Ten przykład pokazuje ograniczenie dopasowania semantycznego dla rzadkich, dokładnych terminów technicznych. Nie dowodzi jednak przewagi jednej metody dla dowolnych zapytań.

### Pytania bez danych

Przy progu podobieństwa **0.38** oba pytania zostały prawidłowo odrzucone: Q11 (wynik 0.292) i Q12 (0.168). Lokalna skuteczność odrzucenia: **2/2**.

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
- Próg odrzucenia 0.38 zweryfikowano lokalnie tylko na dwóch pytaniach bez odpowiedzi.
- Metryki retrieval nie oceniają prawdziwości ani jakości końcowej odpowiedzi LLM, ponieważ model generatywny nie jest podłączony.
- Możliwe rozwinięcia: większy zbiór ground truth, niezależny zbiór testowy, analiza czasu odpowiedzi i kosztu, reranker cross-encoder oraz generacja odpowiedzi z cytowaniami.

## Powtarzalność

Wyniki podane wyżej pochodzą z lokalnego raportu `data/eval_results.json` po poprawkach rankingu agenta. Przed porównywaniem kolejnych zmian należy zachować wersję danych, konfigurację indeksu, próg odrzucenia i ten sam zestaw pytań.
