# Faithfulness — audyt twierdzeń Gemini (weryfikacja asystowana, do potwierdzenia)

**Status:** wstępna ocena tekstowa wykonana z pomocą ChatGPT, **jeszcze nie niezależna ręczna ocena człowieka**. Przed użyciem określenia „human-verified” w README użytkownik powinien przejrzeć adnotacje i zatwierdzić cytaty.

## Wynik

- Model: `google:gemini-3.5-flash-lite`.
- Próba: **5** znanych pytań z odpowiedzią (`data/eval_questions.json`), **nie niezależny holdout**.
- Odpowiedzi: **5 wygenerowanych**.
- Sprawdzalne twierdzenia w pliku audytu: **8**.
- Pełne potwierdzenie w przywołanym źródle: **7**.
- Niepełne potwierdzenie / nieuzasadniona część twierdzenia: **1**.
- Wstępny faithfulness: **7/8 = 87,5%**; jedna cała wypowiedź złożona otrzymuje 0, jeśli choć jeden element jest niepoparty wskazanym cytatem.

## Sprawdzenie każdego twierdzenia

| Pytanie i twierdzenie | Treść | Źródło | Ocena |
|---|---|---|---|
| Q001 · C1 | MBR: EN-DE, EN-RU, PaLM 8B | SOURCE 1 | SUPPORTED |
| Q002 · C1 | Gradual stacking: opis metody | SOURCE 1 | SUPPORTED |
| Q002 · C2 | MIDAS: do 40% szybszy trening — deklaracja autorów | SOURCE 1 | SUPPORTED |
| Q002 · C3 | Zarzut recenzenta: 1B/2B vs 7B/13B | SOURCE 1 | SUPPORTED |
| Q003 · C1 | BoT: drzewa binarne i oryginalność | SOURCE 1 | SUPPORTED |
| Q004 · C1 | KoLA: ROUGE-L, równanie 3 | SOURCE 1 | SUPPORTED |
| Q005 · C1 | PEN/PERM: konieczność i porównanie do wcześniejszych prac | SOURCE 1 | UNSUPPORTED (częściowo trafne) |
| Q005 · C2 | PEN/PERM: nowe zadania vs prostsze zadania wcześniejsze | SOURCE 4 | SUPPORTED |

## Przypadek błędnej nadinterpretacji — Q005, C1

Model: „A reviewer questions the necessity of the PEN and PERM datasets and asks what specific insights or value these two datasets bring **compared to prior work**.”

Cytowana [recenzja 9DIRWfvxxd](https://openreview.net/forum?id=x7AD0343Jz&noteId=9DIRWfvxxd) pyta, **dlaczego te zadania są potrzebne i co wnoszą**, ale nie stawia wprost porównania „compared to prior work”. Takie sformułowanie występuje dopiero w [innej recenzji 0txzQgon0j](https://openreview.net/forum?id=x7AD0343Jz&noteId=0txzQgon0j), przywołanej osobno jako SOURCE 4 dla kolejnego twierdzenia. Gdy oceniamy zgodność **każdego twierdzenia z jego cytatami**, ocena Q005/C1 to `unsupported` (częściowa zgodność nie wystarczy).

**Możliwa poprawiona wypowiedź:** „A reviewer questions why the PEN and PERM datasets are needed and what insights they offer.” — poparta SOURCE 1, bez nieuzasadnionego dopowiedzenia.

## Ważny niuans — Q002, C3

W SOURCE 1 recenzent krytykuje ograniczenie eksperymentów do modeli 1B i 2B i pyta o skalowalność. Abstrakt autorów tej samej publikacji wymienia jednak również model 8B. Twierdzenie modelu jest ocenione jako `supported`, **ponieważ przypisuje ten zarzut recenzentowi**; nie oznacza to, że cały artykuł faktycznie nie zawiera eksperymentów na 8B. Warto zachować to rozróżnienie w syntezie.

## Metodologia i ograniczenia

1. Każdy wpis `claims[]` oceniamy oddzielnie względem **wyłącznie jego `cited_source_numbers`**, z rozróżnieniem `author_abstract` i pól recenzji.
2. Wszystkie części twierdzenia muszą mieć oparcie w przywołanym materiale; samo istnienie URL lub numeru cytowania nie wystarcza.
3. Cytaty z odpowiednich pól, adnotacje i pełny kontekst wejściowy znajdują się w `faithfulness_manual_audit_reviewed.json`.
4. Wyniki obejmują tylko pięć wcześniej znanych pytań i pojedynczy przebieg generowania; nie jest to benchmark niezależny, miara uogólnienia ani dowód zerowej halucynacji.
5. Ta ocena pochodzi od asystenta AI i **wymaga świadomego przeglądu przez autora projektu**, zanim zostanie przedstawiona jako ręcznie zweryfikowana.

### Tekst do README po zatwierdzeniu przez człowieka

> On 5 known answerable questions, Gemini 3.5 Flash-Lite generated 8 annotated factual claims. After claim-level review against each claim's cited OpenReview abstract/review fields, 7/8 claims were fully supported (87.5%); 1/8 included an unsupported comparative qualifier. This is a small, non-held-out sample. Citation format validation is separate from semantic faithfulness, and the result should not be interpreted as generalization performance.
