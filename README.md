# Asystent przeglądu literatury — RAG + agent + Gemini

**OpenReview · ICLR 2024 + NeurIPS 2024 · ChromaDB · BM25 / RRF · PydanticAI · Streamlit**

Asystent wspomagający przegląd publikacji ML: wyszukuje prace i ich **oficjalne recenzje**, zestawia deklaracje autorów z uwagami recenzentów oraz wskazuje oryginalne źródła w OpenReview. Opcjonalnie generuje odpowiedź przez **Gemini** z użyciem PydanticAI. Bez klucza API nadal działa pełny tryb *retrieval-only*.

> **Co system robi, a czego nie:** agent jest deterministycznym kontrolerem narzędzi (maksymalnie 3 kroki), nie autonomicznym agentem LLM. Generowanie jest osobną, opcjonalną warstwą. Walidacja numerów cytowań nie jest dowodem, że każde twierdzenie jest prawdziwe — dlatego projekt zawiera odrębną ocenę faithfulness.

![Widok wyszukiwania i recenzji OpenReview](assets/app-preview.png)

## Najważniejsze wyniki

| Miara | Wynik | Kontekst |
|---|---:|---|
| Dokumenty | **80 publikacji, 308 oficjalnych recenzji** | 40 prac z ICLR 2024 i 40 z NeurIPS 2024 |
| Indeks wektorowy | **1 344 fragmenty recenzji** | Dzielenie świadome limitu tokenów modelu |
| Agent — Review Hit@5 | **100%** | 8 oznaczonych recenzji w głównym zestawie |
| Agent — MRR@5 | **1,000** | Główny zestaw; wynik nie stanowi miary uogólnienia |
| RRF — MRR@5 | **1,000** | Dodatkowe 4 pytania z oznaczonymi recenzjami |
| Faithfulness generacji | **7/8 (87,5%)** | Ocena wspomagana na 5 znanych pytaniach; do ręcznego zatwierdzenia |
| Testy | **50 + 6** | 50 testów funkcjonalnych i 6 testów audytu faithfulness; oba zestawy zaliczone osobno |

**Dlaczego taki zakres?** Celem jest odtwarzalna analiza małego, kontrolowanego korpusu, a nie maksymalizacja liczby dokumentów. Rzetelne uzasadnienie wyników i błędów jest ważniejsze od pozornie idealnych metryk.

## Jak działa system

```text
Pytanie użytkownika (+ opcjonalny filtr konferencji)
          │
          ▼
Wyszukiwanie: ChromaDB / BM25 / hybryda RRF
          │
          ▼
Agent deterministyczny (maks. 3 kroki, trace)
  ├─ przeformułowanie pytania i wyszukiwanie hybrydowe
  ├─ porządkowanie według terminów technicznych
  ├─ pobranie innych recenzji tej samej pracy (suplement)
  └─ abstention: próg wektorowy + kontrola nazw modeli
          │
          ▼
Źródła: recenzje + osobne abstrakty autorów + linki
          │
          ├─────────────► Retrieval-only (bez API)
          │
          ▼
PydanticAI + Gemini (opcjonalnie, po zgodzie użytkownika)
          │
          ▼
Ustrukturyzowana odpowiedź + kontrola cytowań
```

Przechowywany jest **prompt z kontekstem przekazanym modelowi**, lista źródeł i przebieg kroków agenta. Suplementarne recenzje nie są liczone jako osobne wyniki rankingu. Autorom przypisujemy wyłącznie treści abstraktów; oceny i krytykę — recenzentom.

### Dane i indeks

- **Źródło:** OpenReview API v2, `data/papers.json` i `data/reviews.json`. Reguły doboru i moment pobrania próby: `data/sampling_metadata.json`.
- **Jednostka wyszukiwania:** fragment recenzji w ChromaDB (SentenceTransformer `all-MiniLM-L6-v2`), scalany do unikalnego `review_id` według najlepszego fragmentu. Oryginalne pola recenzji pozostają nieucięte w źródle i kontekście odpowiedzi.
- **Chunking:** okno do 192 tokenów, nakładanie 32 tokenów. Z 308 recenzji powstały 1 344 fragmenty. Rozwiązuje problem przekraczania limitu sekwencji modelu embeddingowego przez pełne recenzje.
- **BM25:** pełna recenzja razem z abstraktem w głównej implementacji (oddzielna ablacja rozdziela te składniki); **RRF:** łączenie rankingów. Indeks abstraktowy ma dokładnie jeden dokument na pracę.
- **Aktualność indeksu:** fingerprint treści i modelu, automatyczna przebudowa przy zmianie danych lub konfiguracji.

### Evidence Gate i bezpieczne odpowiedzi

Agent może odmówić odpowiedzi, gdy dane są niewystarczające. Oprócz progu podobieństwa wektorowego (`0.38`) stosuje konserwatywne sprawdzenie, czy konkretne nazwy wersji modeli wymienione w pytaniu występują łącznie w którymś ze znalezionych źródeł. Normalizacja obsługuje np. `PaLM8B` / `PaLM 8B`. **To kontrola obecności nazw, a nie pełne rozumienie, czy źródło dowodzi tezy.**

Dla Q106 pytającego o porównanie zużycia energii GPT-6 i Claude 5 wynik wektorowy był relatywnie wysoki (`0,480`), ale recenzje dotyczyły innych modeli; agent po poprawce odmawia odpowiedzi. Ten przypadek **został użyty do opracowania reguły**, dlatego nie jest niezależnym testem jej skuteczności.

## Uruchomienie

**Wymagania:** Git, Python **3.12** (zalecany), dostęp do internetu podczas pierwszego pobrania modelu embeddingowego. Przetestowano także macOS z Pythonem 3.14. Python 3.15 nie był weryfikowany. **Tryb offline nie wymaga klucza Gemini.**

Poniższe instrukcje wykonuj **po kolei, po jednej komendzie**. Najpierw opisano Windows, a następnie macOS/Linux. Nie kopiuj tekstów objaśnień do terminala.

### Windows — PowerShell (zalecana ścieżka)

**1. Przejdź do katalogu użytkownika** (nie uruchamiaj `git clone` w `C:\Windows\System32`):

```powershell
cd $HOME
```

**2. Pobierz repozytorium:**

```powershell
git clone https://github.com/maciejonisiewicz/solvro-rag-rekrutacja.git
```

**3. Wejdź do projektu:**

```powershell
cd solvro-rag-rekrutacja
```

**4. Utwórz środowisko Python 3.12:**

```powershell
py -3.12 -m venv .venv
```

**5. Zainstaluj biblioteki:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**6. Uruchom aplikację:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Polecenia powyżej **nie wymagają aktywacji `.venv`**, więc działają również wtedy, gdy PowerShell blokuje skrypty `*.ps1`. Uruchamiaj je z katalogu projektu.

**Opcjonalnie: aktywacja środowiska.** Jeśli chcesz używać krótszych poleceń `python ...`, możesz aktywować środowisko:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jeżeli PowerShell wyświetli `running scripts is disabled on this system`, **nie musisz zmieniać zabezpieczeń** — używaj poleceń z `.\.venv\Scripts\python.exe` pokazanych wyżej. Alternatywnie, aby odblokować lokalną aktywację **tylko w obecnym oknie PowerShell**, wykonaj:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
```

A potem ponownie uruchom `Activate.ps1`. Ustawienie dla `Process` jest tymczasowe; w środowisku z politykami firmowymi może być nadal blokowane.

### macOS / Linux — Terminal (zsh lub Bash)

**1. Przejdź do katalogu użytkownika:**

```bash
cd ~
```

**2. Pobierz repozytorium:**

```bash
git clone https://github.com/maciejonisiewicz/solvro-rag-rekrutacja.git
```

**3. Wejdź do projektu:**

```bash
cd solvro-rag-rekrutacja
```

**4. Utwórz środowisko Python:**

```bash
python3 -m venv .venv
```

**5. Aktywuj środowisko:**

```bash
source .venv/bin/activate
```

**6. Zainstaluj biblioteki:**

```bash
python -m pip install -r requirements.txt
```

**7. Uruchom aplikację:**

```bash
python -m streamlit run app.py
```

### Co zobaczysz po uruchomieniu?

Aplikacja działa lokalnie pod **http://localhost:8501**. Przy pierwszym wyszukiwaniu może pobrać model embeddingowy i automatycznie utworzyć indeks ChromaDB w `data/chroma_db/`. Nie zamykaj terminala, w którym działa Streamlit; aby zakończyć pracę, naciśnij `Ctrl+C`.

Główna strona **app** zawiera wyszukiwarkę RAG i panel ewaluacji. W lewym menu wybierz **Gemini Asystent**. Dostępne tryby: **Tylko źródła (offline)** oraz **Gemini — odpowiedź z cytowaniami**.

### Opcjonalna integracja PydanticAI + Gemini

**Bez klucza API** wyszukiwanie i tryb offline nadal działają. Generowanie odpowiedzi jest opcjonalne i może wykorzystać limit lub spowodować opłatę u dostawcy.

**Windows PowerShell — doinstaluj bibliotekę:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-llm.txt
```

**macOS / Linux — doinstaluj bibliotekę** (w aktywnym `.venv`):

```bash
python -m pip install -r requirements-llm.txt
```

W [Google AI Studio](https://aistudio.google.com/api-keys) wygeneruj swój klucz API. **Nie wklejaj klucza do kodu, commita, pliku README ani komunikatora.**

**Windows PowerShell — ustaw klucz bez jego wyświetlania:**

```powershell
$env:GOOGLE_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Wklej klucz Google Gemini' -AsSecureString)).Password
```

**macOS zsh — wpisz klucz interaktywnie:**

```zsh
read -rs 'GOOGLE_API_KEY?Wklej klucz Google Gemini: '
```

Następnie, w tym samym terminalu:

```zsh
export GOOGLE_API_KEY
```

**Linux Bash — wpisz klucz interaktywnie:**

```bash
read -r -s -p 'Wklej klucz Google Gemini: ' GOOGLE_API_KEY
```

Zakończ ukryte wprowadzanie nową linią:

```bash
echo
```

A następnie udostępnij zmienną programowi:

```bash
export GOOGLE_API_KEY
```

**Test generowania z terminala — Windows:**

```powershell
.\.venv\Scripts\python.exe llm_synthesis.py --query "Which review criticizes ROUGE-L in Equation 3?" --model google:gemini-3.5-flash-lite
```

**Test generowania z terminala — macOS / Linux:**

```bash
python llm_synthesis.py --query "Which review criticizes ROUGE-L in Equation 3?" --model google:gemini-3.5-flash-lite
```

Oczekiwany wynik testu przy dostępnym API: `STATUS: generated`, recenzja `2mTTJZa647` publikacji KoLA oraz cytowanie `[SOURCE 1]`. Dostępność modelu i opłaty zależą od konta. Bez klucza API polecenie powinno przejść w `retrieval_only`: zachowuje wyniki wyszukiwania i kontekst bez udawania odpowiedzi; błędy dostawcy również nie usuwają źródeł.

**Test generowania w Streamlit:** jeśli aplikacja była uruchomiona **przed** ustawieniem `GOOGLE_API_KEY`, zatrzymaj ją (`Ctrl+C`) i uruchom ponownie **z tego samego terminala, w którym ustawiono klucz**. Potem wybierz **Gemini Asystent → Gemini — odpowiedź z cytowaniami**. Aby sprawdzić tryb offline, wybierz **Tylko źródła (offline)** — klucz nie jest potrzebny.

## Ewaluacja retrieval

### Metodologia

- **Zestaw główny:** `data/eval_questions.json` — 10 pytań z odpowiedzią oraz 2 bez wystarczających danych; w 8 pytaniach z odpowiedzią wskazano ręcznie konkretną recenzję.
- **Dodatkowy zestaw:** `data/eval_holdout_verified.json` — 4 pytania z oznaczoną recenzją i 2 pytania bez danych.
- **Paper Hit@5** liczymy na pytaniach z odpowiedzią, na poziomie publikacji. **Review Hit@5, MRR@5 i nDCG@5** liczymy tylko tam, gdzie wskazano `target_review_id`; fragmenty tej samej recenzji są deduplikowane. W nDCG oceniona jest jedna recenzja (`relevance=1`), a nieoznaczone recenzje mają 0 **w metryce**, niekoniecznie są rzeczywiście nietrafne.
- Wariant „Tylko abstrakty” ma tylko Paper Hit@5; recenzji w nim nie ma.
- Heurystyki agenta były rozwijane ze znajomością małego zestawu głównego. Dodatkowy zbiór wykorzystano później również do diagnozy błędu Q106, więc nie jest już czystym holdoutem dla końcowego Evidence Gate.

### Wyniki po chunkingu — zestaw główny

| Metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Wektorowe — ChromaDB | 90% | 87,5% | 0,656 | 0,712 |
| BM25 — recenzje + abstrakty | 100% | 100% | 0,917 | 0,938 |
| Hybrydowe — RRF | 100% | 100% | 0,906 | 0,929 |
| Agent deterministyczny | **100%** | **100%** | **1,000** | **1,000** |
| Tylko abstrakty | 90% | N/A | N/A | N/A |

### Wyniki po chunkingu — dodatkowy zestaw (4 recenzje oznaczone)

| Metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| Wektorowe — ChromaDB | 100% | 100% | 0,875 | 0,908 |
| BM25 — recenzje + abstrakty | 100% | 100% | 0,875 | 0,908 |
| Hybrydowe — RRF | 100% | 100% | **1,000** | **1,000** |
| Agent deterministyczny | 100% | 100% | **1,000** | **1,000** |
| Tylko abstrakty | 100% | N/A | N/A | N/A |

Powyższe tabele dotyczą **wyszukiwania**, a nie jakości generacji ani poprawności końcowego odrzucenia odpowiedzi. Raporty: `data/eval_results.json`, `data/eval_holdout_results.json` oraz zachowane migawki w `data/experiments/`.

### Ablacja: czy BM25 ma nieuczciwą przewagę dzięki abstraktom?

Skrypt `retrieval_ablation.py` porównuje pełne recenzje oraz **te same fragmenty wejściowe** w BM25 i ChromaDB (główny, znany zestaw pytań; nie jest to nowy holdout):

| Korpus i metoda | Paper Hit@5 | Review Hit@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25, recenzje + abstrakty | 100% | 100% | 0,917 | 0,938 |
| BM25, tylko pełne recenzje | 100% | 100% | 0,917 | 0,938 |
| BM25, te same chunki co model wektorowy | 100% | 100% | 0,838 | 0,877 |
| ChromaDB, te same chunki | 90% | 87,5% | 0,656 | 0,712 |

Abstrakty **nie zmieniły wyniku** BM25 na tym zestawie. Metoda leksykalna nadal ma lepsze MRR od semantycznej przy tej samej treści fragmentów. To jednak nie jest całkowicie identyczna konfiguracja: tokenizer, sposób punktowania i wyszukiwanie przybliżone mogą się różnić. Nie wnioskujemy o uniwersalnej przewadze BM25. Wyniki: `data/experiments/fair_corpus_ablation.json`.

**Analiza błędu:** w Q002 właściwa recenzja była dopiero na **8. pozycji wektorowej**, a na 1. pozycji BM25, RRF i agenta. Nazwy i rozmiary modeli oraz inne dokładne sformułowania mogą być skuteczniej wychwytywane przez metodę leksykalną.

### Odrzucanie pytań bez wystarczających danych

Należy odróżnić dwie oceny:

1. **Sam próg podobieństwa wektorowego `0,38`:** główny zestaw odrzuca 2/2 pytań bez danych; dodatkowy zestaw — 1/2 (Q106 ma score `0,480` i nie jest odrzucone). To diagnostyka progu, nie końcowa decyzja agenta.
2. **Decyzja agenta po Evidence Gate:** główny zestaw: 10/10 pytań z odpowiedzią zaakceptowanych i 2/2 pytań bez danych odrzuconych; dodatkowy zestaw: 4/4 i 2/2. Drugi wynik to **diagnostyka po wykorzystaniu Q106 w rozwoju reguły**, nie niezależna walidacja.

## Faithfulness — jakość wygenerowanych odpowiedzi

Na 5 pytaniach ze znanego zbioru wygenerowano 5 odpowiedzi `google:gemini-3.5-flash-lite`; podzielono je na 8 sprawdzalnych twierdzeń. Każde porównano z **dokładnie cytowaną recenzją lub abstraktem** — pełne potwierdzenie całej wypowiedzi wymagane jest również dla zdań złożonych.

| Ocena | Liczba |
|---|---:|
| Potwierdzone przez cytowane źródło | **7** |
| Niepotwierdzone w pełni | **1** |
| Niejednoznaczne | **0** |
| **Wstępny faithfulness** | **7/8 = 87,5%** |

Jest to **wstępna adnotacja przygotowana z pomocą asystenta, wymagająca zatwierdzenia przez autora** — nie deklarujemy niezależnej, ręcznej oceny człowieka. Zestaw jest mały i znany systemowi, więc wynik nie może być uogólniany.

Przykład błędu: jedna odpowiedź o zbiorach PEN/PERM poprawnie opisuje krytykę recenzenta, ale dodaje porównanie do wcześniejszych badań, którego **wskazana** recenzja nie potwierdza wprost. Link i numer cytowania były prawidłowe, lecz twierdzenie otrzymało ocenę `unsupported`.

Szczegółowa metodologia, dowody i ograniczenia: [`docs/faithfulness_review_report.md`](docs/faithfulness_review_report.md). Adnotacje: `data/experiments/faithfulness_manual_audit_reviewed.json`.

## Uruchamianie testów i eksperymentów

Poniższe polecenia uruchamiaj z katalogu projektu w aktywnym środowisku `.venv`. Na Windowsie, **jeżeli nie aktywujesz środowiska**, zamień początkowe `python` na `.\.venv\Scripts\python.exe` (bez zmiany reszty polecenia).

**Testy automatyczne (offline):**

```bash
python -m unittest -v test_evaluate test_rag_index test_context_integrity test_chunking test_retrieval_ablation test_evidence_gate test_llm_synthesis test_streamlit_llm_bridge test_faithfulness_collect
```

**Raporty ewaluacji** — uruchomienie może nadpisać bieżące pliki wynikowe:

```bash
python evaluate.py
```

```bash
python holdout_eval.py --run
```

**Porównanie metod przy kontrolowanej treści recenzji:**

```bash
python retrieval_ablation.py
```

**Audyt decyzji agenta — zestaw główny:**

```bash
python agent_abstention_audit.py --questions data/eval_questions.json
```

**Audyt decyzji agenta — zestaw dodatkowy:**

```bash
python agent_abstention_audit.py --questions data/eval_holdout_verified.json
```

**Zliczenie istniejących adnotacji faithfulness** (bez wywoływania Gemini):

```bash
python faithfulness_collect.py score --file data/experiments/faithfulness_manual_audit_reviewed.json
```

Nowa generacja do audytu (`faithfulness_collect.py collect`) wymaga klucza API, może być płatna i może dać inne wyniki niż zapisana próba; nie nadpisuj adnotacji bez zachowania poprzedniej wersji.

## Najważniejsze pliki

```text
app.py                         # Główna wyszukiwarka + panel metryk
pages/01_Gemini_Asystent.py    # Opcjonalna strona generująca odpowiedzi
streamlit_llm_bridge.py        # Połączenie Streamlit z agentem i syntezą
rag_system.py                  # ChromaDB, BM25, RRF, chunking
agent_rag.py                   # Deterministyczny agent, trace i prompt
evidence_gate.py               # Konserwatywny filtr nazw modeli
llm_synthesis.py               # PydanticAI i walidacja cytowań
fetch_data.py                  # Pobieranie OpenReview
evaluate.py                    # Paper/review Hit@5, MRR, nDCG
holdout_eval.py                # Dodatkowy zestaw pytań
retrieval_ablation.py          # Kontrolowane porównanie korpusów
agent_abstention_audit.py      # Audyt decyzji agent/abstention
faithfulness_collect.py        # Zbieranie i ocenianie twierdzeń
requirements.txt               # Zależności bazowe
requirements-llm.txt           # Opcjonalne zależności PydanticAI
docs/faithfulness_review_report.md

data/papers.json, data/reviews.json, data/sampling_metadata.json
data/eval_questions.json, data/eval_holdout_verified.json
data/eval_results.json, data/eval_holdout_results.json
data/experiments/fair_corpus_ablation.json
data/experiments/faithfulness_manual_audit_reviewed.json
```

## Ograniczenia i uczciwa interpretacja

- Mało pytań i zwykle tylko jedna oznaczona recenzja na pytanie; nieoznaczony dokument może również zawierać trafną odpowiedź.
- Wyniki agenta na zbiorze głównym mogą być zawyżone przez rozwój heurystyk z jego znajomością; Q106 wykorzystano do budowy Evidence Gate. Przed mocniejszymi wnioskami wymagany jest nowy, niezależny zestaw.
- Sprawdzanie nazw konkretnych modeli może odrzucić dobre odpowiedzi (false negative), a współwystępowanie nazw **nie gwarantuje** poparcia dla twierdzenia.
- BM25 i embeddingi używają różnej tokenizacji / funkcji scoringowej. Ablacja kontroluje treść wejścia, nie wszystkie czynniki.
- Cytowanie poprawnego dokumentu nie dowodzi faithfulness. Wstępny audyt 7/8 ma ograniczony zakres i wymaga zatwierdzenia człowieka.
- Działanie Gemini, dostęp do określonych modeli, koszty i limity zależą od zewnętrznego dostawcy. Gdy API jest niedostępne, pozostaje tryb retrieval-only.

## Powtarzalność i bezpieczeństwo

Dane OpenReview oraz pytania/etykiety nie są dokumentami indeksowanymi razem z odpowiedziami wzorcowymi. Wyniki i eksperymenty zachowują identyfikatory publikacji i recenzji. Po zmianie danych lub modelu indeks jest przebudowywany na podstawie fingerprintu. Nie commituj kluczy API, plików `.env`, bazy ChromaDB ani bieżących logów sesji. Aby wiernie porównać wyniki, zachowaj ten sam korpus, oznaczenia, model embeddingowy, chunking i wersje kodu.
