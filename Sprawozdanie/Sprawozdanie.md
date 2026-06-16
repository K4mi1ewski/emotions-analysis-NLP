# Sprawozdanie: Emocjonalna mapa języka — analiza emocji w tekstach kultury

**Autorzy:**

- Kamil Milewski
- Patryk Fergisz
- Marcin Rykała

---

## 1. Opis problemu i celu projektu
Rozpoznawanie emocji w tekstach stanowi jedno z kluczowych, a zarazem najtrudniejszych zagadnień we współczesnym przetwarzaniu języka naturalnego. W odróżnieniu od klasycznej analizy sentymentu ograniczającej się do klasyfikacji tekstu w trzech wymiarach - pozytywny, negatywny, neutralny - analiza emocji wymaga od modeli sztucznych sieci neuronowych zrozumienia znacznie bardziej złożonych niuansów semantycznych. Ich celem jest identyfikacja konkretnych, zróżnicowanych stanów, takich jak np. radość, smutek, złość, strach, podziw.

Głównym celem projektu jest przeprowadzenie analizy emocji w wypowiedziach poprzez wytrenowanie oraz zastosowanie zaawansowanych modeli sztucznych sieci neuronowych. Oznacza to zbadanie rozkładu i dominacji poszczególnych stanów emocjonalnych w wykorzystanym tekście oraz określenie, z jaką precyzją nowoczesne architektury NLP potrafią te stany mapować.

Porównaliśmy dwa podejścia do klasyfikacji emocji:

1. Podejście z wykorzystaniem fine-tuningu modeli opartych na architekturze Enkodera z rodziny BERT (konkretnie: RoBERTa, DistilBERT, ALBERT). Podejście to wymaga dostosowania wag wytrenowanych językowo modeli na odpowiednio przygotowanym zbiorze uczącym, co pozwala modelom na wyuczenie się charakterystyki 28 zdefiniowanych klas emocjonalnych.
2. Podejście z wykorzystaniem generatywnych, dużych modeli językowych (LLM) w architekturze Dekodera. Celem tego podejścia jest zbadanie wrodzonych zdolności modeli generatywnych do rozpoznawania i etykietowania emocji wyłącznie na podstawie zapytania, bez kosztownego procesu douczania. Ze względu na narzut obliczeniowy, modele GPT testowaliśmy wyłącznie na części zbioru danych.

---

## 2. Opis wykorzystanego zbioru danych

W projekcie wykorzystano zbiór danych GoEmotions (`google-research-datasets/go_emotions`) udostępniony przez Google Research (w konfiguracji `simplified`). Jest to zbiór anglojęzycznych komentarzy pochodzących z platformy Reddit, które zostały ręcznie wyetykietowane pod kątem występowania w nich różnych emocji.

Oryginalnie zbiór ten jest zbiorem wieloetykietowym, co oznacza, że jeden tekst może wyrażać kilka emocji jednocześnie. W celu uproszczenia zadania klasyfikacji i sprowadzenia go do standardowego problemu jednoetykietowego, zbiór został poddany wstępnej filtracji. Z danych odrzucono wszystkie przykłady posiadające więcej niż jedną przypisaną emocję, a jedyną pozostawioną etykietę wyodrębniono jako docelową zmienną objaśnianą.

Po zastosowaniu filtru wykluczającego teksty o wielu etykietach, rozmiary poszczególnych podzbiorów ukształtowały się następująco:
* Zbiór treningowy: 36 308 przykładów
* Zbiór walidacyjny: 4 548 przykładów
* Zbiór testowy: 4 590 przykładów

Zbiór zawiera 28 unikalnych klas, na które składa się 27 zdefiniowanych stanów emocjonalnych (np. *admiration*, *amusement*, *anger*, *joy*) oraz jedna klasa *neutral* oznaczająca brak wyraźnego nacechowania emocjonalnego:

`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`.

Największa klasa to `neutral` (12823 w train), najmniejsze to `grief` (39) i `pride` (51), więc zbiór jest mocno niezbalansowany.

Ta dysproporcja pomiędzy poszczególnymi klasami miała bezpośrednie przełożenie na proces uczenia i konieczność doboru metryk ewaluacji.

![Rozklad klas](./rozkladklas.png)

Klasy, w celu późniejszej ewaluacji sentymentu, zostały podzielone na 4 grupy:

![4 klasy sentymentu](./4grupy.png)

---

## 3. Opis zastosowanej architektury sieci neuronowej

Zarówno modele BERT (Bidirectional Encoder Representations from Transformers) jak i GPT (Generative Pre-trained Transformer) są oparte na architekturze transformerów.

### 3.1. Modele z rodziny BERT (Architektura Enkodera)
Modele tego typu przetwarzają tekst dwukierunkowo, co pozwala im na głębokie zrozumienie kontekstu całego zdania. Są dobre do klasyfikacji, my przeprowadziliśmy fine-tuning tych modeli pod konkretne etykiety. W projekcie zaimplementowano trzy warianty z tej rodziny:
1. RoBERTa (Robustly Optimized BERT Approach) – zoptymalizowana wersja BERT-a, trenowana dłużej, na większym korpusie danych i z użyciem dynamicznego maskowania tokenów, posiadająca większy słownik niż BERT (ok. 500 MB)
2. ALBERT (A Lite BERT) – odchudzona architektura projektowa wykorzystująca techniki współdzielenia parametrów między warstwami - najlżejszy model (ok. 47 MB)
3. DistilBERT – wersja BERT-a wykorzystująca proces destylacji wiedzy. Posiada znacznie mniej parametrów, przez co charakteryzuje się najniższym czasem wnioskowania przy zachowaniu wysokiej skuteczności (2 razy mniej warstw niż podstawowy BERT - ok. 260 MB)

### 3.2. Modele Generatywne GPT (Architektura Dekodera)
Jako rozszerzenie eksperymentu wykorzystano dwa duże modele językowe z rodziny GPT. W odróżnieniu od BERTów, modele te przetwarzają tekst jednokierunkowo i służą do generowania odpowiedzi na podstawie zadanego kontekstu - promptu.

Zastosowane modele charakteryzują się ogromną liczbą parametrów. Aby umożliwić ich uruchomienie w dostępnym środowisku wykonawczym (ograniczona pamięć VRAM), zastosowano kwantyzację 4-bitową. Jest to technika kompresji modelu, która zmniejsza precyzję wag sieci z 16 lub 32 bitów do 4 bitów, drastycznie redukując zapotrzebowanie na pamięć bez krytycznej utraty zdolności logicznych modelu. Testowanie przeprowadzaliśmy techniką zero-shot - bez żadnego wstępnego przygotowania modeli pod nasze dane. Użyliśmy dwóch, następujących modeli (oba ważące ok. 6-7 GB):

- `unsloth/llama-3-8b-Instruct-bnb-4bit` (Llama 3),
- `unsloth/gemma-2-9b-it-bnb-4bit` (Gemma 2).

---

## 4. Opis procesu uczenia

Proces pozyskiwania gotowych systemów do analizy emocji w tym projekcie przebiegał zupełnie inaczej dla modeli BERT i GPT, co wynika z przyjętej metodologii.

W kodzie została także uwzględniona możliwość wczytania z dysku wcześniej zapisanych modeli BERT oraz wyników ich testowania.

### 4.1. Fine-tuning modeli z rodziny BERT
Modele enkoderowe zostały poddane procesowi douczania z nadzorem na docelowym zbiorze danych GoEmotions. Jako funkcję straty dla wieloklasowej klasyfikacji jednoetykietowej zastosowano Entropię Krzyżową. Algorytmem optymalizującym wagi był standardowy w środowisku NLP i domyślny dla biblioteki Transformers wariant optymalizatora Adam, uwzględniający zanikanie wag.

Ze względu na wysoką złożoność obliczeniową procesu treningowego na dużej próbie tekstów i niestabilność środowiska, proces uczenia nie odbył się w jednym nieprzerwanym ciągu. Zastosowano mechanizm punktów kontrolnych. Modele były trenowane w mniejszych, odrębnych sesjach, a postępy wag były zapisywane na dysku. Po wymuszonym przerwaniu sesji trening był wznawiany od ostatniego zapisanego punktu, natomiast w przypadku ewaluacji końcowej wczytywano najlepsze zapisane wagi do obliczenia metryk na zbiorze testowym. Pozwoliło to zminimalizować ryzyko utraty postępów.

Czas uczenia 3 modeli po 3 epoki każdy wynosił ok. 50 minut na Google Colab T4 GPU.

![Argumenty treningu](./trainargs.png)


### 4.2. Wnioskowanie Zero-Shot dla modeli GPT
Dwa wykorzystane modele generatywne (GPT) nie były poddawane żadnemu procesowi uczenia. Ich implementacja opierała się na podejściu Zero-shot. Oznacza to, że modele otrzymały wyłącznie sformatowane zapytanie wejściowe, w którym zdefiniowano zadanie klasyfikacji tekstu do jednej z 28 dostępnych klas. Odpowiedź modelu była odpowiednio parsowana do jednej z etykiet.

Przeprowadzenie pełnej ewaluacji modeli GPT na całym zbiorze testowym byłoby trudne ze względów wydajnościowych. Ze względu na długi czas generacji odpowiedzi, modele GPT były testowane jedynie na 1000 pierwszych próbek ze zbioru testowego - 2 modele po 1000 próbek generowały odpowiedzi przez około 1 godzinę.

---

## 5. Opis zastosowanych metryk ewaluacji

W projekcie zastosowaliśmy kilka różnych metryk i technik ewaluacji. Jako podstawa do obliczania metryk, po testowaniu zapisywaliśmy następujące wyniki dla każdego modelu:

-`y_true: prawdziwe emocje`

-`pred: przewidywane emocje`

-`proba: rozkłady prawdopodobieństw odpowiedzi (dla modeli BERT)`

-`metrics: accuracy, F1-Macro, F1-Weighted, Cohen's Kappa`

-`extra: latencja dla zapytania`

### 5.1. Dokładność (Accuracy)
Dokładność to najbardziej intuicyjna metryka, zdefiniowana jako stosunek liczby poprawnych predykcji do wszystkich przykładów w zbiorze testowym:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Liczba poprawnych predykcji}}{\text{Wszystkie przykłady}}$$

W przypadku analizowanego zbioru GoEmotions, gdzie klasa neutral stanowi ponad 35% danych, metryka ta jest mało miarodajna. Model, który klasyfikowałby każdy tekst jako neutral, osiągnąłby wysoką dokładność bazową, nie ucząc się zarazem rozpoznawania żadnej z właściwych emocji. Dlatego potraktowano ją jedynie jako metrykę pomocniczą.

### 5.2. Precyzja (Precision) i Czułość (Recall)
Dla każdej z 28 klas metryki te obliczane są indywidualnie, stanowiąc bazę pod bardziej zaawansowane wskaźniki:
* Precyzja (Precision): Określa, jaka część tekstów wskazanych przez model jako dana emocja faktycznie do niej należy. Chroni przed błędami typu False Positive.
$$\text{Precision} = \frac{TP}{TP + FP}$$

* Czułość (Recall): Określa, jaką część wszystkich rzeczywistych przykładów danej emocji model był w stanie poprawnie wykryć. Chroni przed błędami typu False Negative.
$$\text{Recall} = \frac{TP}{TP + FN}$$

### 5.3. Miara F1 (F1-Score) i jej wariant Macro
Miara F1 jest średnią harmoniczną precyzji i czułości. Pozwala na jednoczesne balansowanie obu tych wartości:

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

W zadaniach wieloklasowych kluczowy jest sposób agregacji tego wskaźnika. W projekcie jako kluczową metrykę porównawczą wybrano Macro-F1 Score.

Metoda Macro oblicza miarę F1 niezależnie dla każdej z 28 klas, a następnie wyciąga z nich zwykłą średnią arytmetyczną:

$$\text{Macro-F1} = \frac{1}{N} \sum_{i=1}^{N} \text{F1-Score}_i$$

*gdzie $N = 28$ - liczba klas.*

Wariant Macro traktuje każdą klasę - emocję - z dokładnie taką samą wagą. Niezależnie od tego, czy klasa liczy tysiące przykładów, czy zaledwie kilkadziesiąt, jej wpływ na ostateczny wynik wynosi $\frac{1}{28}$. Dzięki temu niska skuteczność na klasach mniejszościowych drastycznie obniża wynik końcowy, co idealnie pokazuje rzeczywistą jakość działania sieci neuronowej.

### 5.4. Wariant Weighted miary F1 (Weighted-F1)
Weighted-F1 to drugi sposób agregacji miary F1 w zadaniu wieloklasowym. Tak jak Macro, oblicza F1 osobno dla każdej z 28 klas, ale przy uśrednianiu nadaje każdej klasie wagę proporcjonalną do liczby jej rzeczywistych przykładów (wsparcia) w zbiorze testowym:

$$\text{Weighted-F1} = \sum_{i=1}^{N} \frac{n_i}{n} \cdot \text{F1-Score}_i$$

*gdzie $n_i$ - liczba przykładów klasy $i$, $n$ - łączna liczba przykładów, $N = 28$.*

W odróżnieniu od wariantu Macro, Weighted-F1 jest zdominowany przez klasy liczne (w tym przypadku przez klasę neutral), przez co przyjmuje wartości zbliżone do dokładności. Z tego powodu potraktowano go jako miarę uzupełniającą, a nie główne kryterium porównawcze.

### 5.5. Wskaźnik Kappa Cohena (Cohen's Kappa)
Wskaźnik Kappa Cohena mierzy zgodność predykcji modelu z etykietami rzeczywistymi, korygując ją o zgodność, którą można by uzyskać czysto losowo:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

*gdzie $p_o$ - zaobserwowana zgodność (równa dokładności), $p_e$ - zgodność oczekiwana przy losowym zgadywaniu zgodnym z rozkładem klas.*

Wartość $\kappa = 1$ oznacza pełną zgodność, $\kappa = 0$ zgodność na poziomie losowania, a wartości ujemne zgodność gorszą od losowej. W przeciwieństwie do samej dokładności, Kappa nie nagradza modelu za przewidywanie wyłącznie dominującej klasy neutral, dlatego przy tak niezbalansowanym zbiorze jest bardziej miarodajnym wskaźnikiem rzeczywistej jakości.

### 5.6. Macierz Pomyłek (Confusion Matrix)
Macierz pomyłek to dwuwymiarowa tabela o wymiarach $28 \times 28$, w której osie reprezentują klasy rzeczywiste oraz klasy przewidziane przez model. 

Jest to najważniejsze narzędzie diagnostyczne w projekcie, pozwalające na wizualną identyfikację, które emocje są ze sobą najczęściej mylone przez sieć, a takze w jakim stopniu dominująca klasa neutral przyciąga predykcje z innych, rzadszych klas emocjonalnych.

Prezentowane macierze zostały znormalizowane wierszami (każdy wiersz sumuje się do 1, pokazując rozkład predykcji w obrębie danej klasy rzeczywistej). Normalizacja była konieczna ze względu na silne niezbalansowanie klas - na surowych, bezwzględnych zliczeniach liczne klasy całkowicie przytłaczałyby rzadkie, czyniąc wykres nieczytelnym. Dzięki normalizacji skuteczność na klasach mniejszościowych jest porównywalna wizualnie z klasami licznymi.

### 5.7. Test McNemara
Test McNemara to nieparametryczny test statystyczny służący do sprawdzenia, czy różnica w trafności dwóch modeli na tym samym zbiorze przykładów jest istotna statystycznie, czy też wynika z przypadku. Opiera się on na przypadkach, w których modele się różnią - tj. na liczbie przykładów, które poprawnie zaklasyfikował tylko pierwszy model ($b$), oraz tych, które poprawnie zaklasyfikował tylko drugi ($c$):

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

Niska wartość $p$ (przyjęto próg $p < 0.05$) oznacza, że różnica między modelami jest istotna statystycznie. W projekcie test zastosowano do porównań par modeli na wspólnym podzbiorze testowym (wspólnym dla modeli BERT i GPT, ze względu na ograniczenie ewaluacji GPT do 1000 przykładów).

---

## 6. Przedstawienie i omówienie wyników

W tej sekcji przedstawiono szczegółowe wyniki projektu.

### 6.1. Główne wyniki klasyfikacji (Accuracy i Macro-F1)

Ewaluacja modeli w zadaniu klasyfikacji do 28 etykiet ukazała bardzo wyraźną przewagę modeli z rodziny BERT, które zostały poddane procesowi fine-tuningu.

**Zestawienie metryk wszystkich modeli**

![](./metryki.png)

| Model | Rodzina | Dokładność (Accuracy) | Macro-F1 | Wskaźnik Cohena (Kappa) |
| :--- | :---: | :---: | :---: | :---: |
| **RoBERTa** | BERT | 0.6198 | **0.5228** | 0.5555 |
| **ALBERT** | BERT | **0.6355** | 0.4909 | 0.5585 |
| **DistilBERT** | BERT | 0.6296 | 0.4835 | **0.5609** |
| **Gemma 2** (4-bit) | GPT | 0.2390 | 0.2341 | 0.1921 |
| **Llama 3** (4-bit) | GPT | 0.2270 | 0.1970 | 0.1788 |

*Tabela 1 Zestawienie głównych metryk skuteczności dla wszystkich modeli.*

Najwyższą wartość kluczowej dla tego projektu metryki Macro-F1 - (0.5228) - osiągnął model RoBERTa. To jednak model ALBERT osiągnął nieznacznie wyższą dokładność ogólną (Accuracy: 0.6355), jednak odbyło się to kosztem niższej skuteczności na klasach mniejszościowych. Wynika to bezpośrednio z własności wykorzystanego zbioru danych – ALBERT częściej i bezpieczniej przewidywał klasę dominującą, co sztucznie zawyżyło jego dokładność kosztem zbalansowanej precyzji.

Modele generatywne poradziły sobie z tym zadaniem dosyć słabo (Macro-F1 poniżej 0.24). Rozpoznawanie 28 subtelnych stanów emocjonalnych bez uprzedniego douczania okazało się zadaniem zbyt trudnym dla dużych modeli językowych w architekturze dekodera.

### 6.2 Macierze pomyłek

**Macierz pomyłek — RoBERTa**

![](./matrix-roberta.png)

**Macierz pomyłek — ALBERT**

![](./matrix-albert.png)

**Macierz pomyłek — DistilBERT**

![](./matrix-distilbert.png)

**Macierz pomyłek — Llama 3**

![](./matrix-llama3.png)

**Macierz pomyłek — Gemma 2**

![](./matrix-gemma2.png)

Z powodu bardzo silnego niezbalansowania danych treningowych, modele BERT wykształciły wyraźną tendencję, by w przypadku niepewności klasyfikować tekst jako neutral.
Można zauważyć wyraźny, pionowy pas ciemniejszego koloru dla przewidywanej etykiety neutral. Emocje słabiej reprezentowane w zbiorze były systematycznie ignorowane przez te modele i wchłaniane przez klasę dominującą. W przypadku modeli GPT ten pas nie jest aż tak widoczny, jednak tutaj dużo częściej występowały pomyłki między innymi klasami. Szczególnie można zauważyć bardzo często mylone ze sobą:
-`grief-anger, pride-admiration, neutral-nervousness` dla Llama3
-`pride-amusement, neutral-nervousness` dla Gemma2

W przypadku modeli BERT, przekątna na macierzy rysuje się dużo wyraźniej, co sugeruje większą efektywność.

### 6.3. F1-score per model

Poniższe wykresy prezentują wykresy F1-score dla każdej z 28 emocji osobno, z zaznaczoną średnią (Macro F1):

**F1-score — RoBERTa**

![](./f1roberta.png)

**F1-score — ALBERT**

![](./f1albert.png)

**F1-score — DistilBERT**

![](./f1distilbert.png)

**F1-score — Llama 3**

![](./f1llama3.png)

**F1-score — Gemma 2**

![](./f1gemma2.png)

RoBERTa ma najwyższy średni wynik i lepszą stabilność niż pozostałe modele. ALBERT traci więcej na rzadkich klasach, podobnie jak DistilBERT. Llama3 i Gemma2 osiągnęły dużo niższy wynik niż modele BERT, chociaż wynik Gemma2 jest nieznacznie lepszy.

### 6.4 Top-3 accuracy

Aby pogłębić analizę wyników dla najskuteczniejszej rodziny modeli (BERT), wprowadzono metrykę Top-3 Accuracy. Mierzy ona odsetek przypadków, w których rzeczywista emocja znalazła się w trójce najbardziej prawdopodobnych predykcji sieci.

**Top-3 Accuracy — wszystkie modele**

![](./top3acc.png)

Dla modeli BERT Top-3 jest wysokie (`~0.86-0.87`), a nawet przy błędnym Top-1 poprawna klasa często znajduje się w trzech najlepszych predykcjach - oznacza to że w wielu przypadkach (ok. 20%) mimo błędnej predykcji ostatecznej, model był całkiem blisko poprawnej klasyfikacji.

### 6.5 Cohen's kappa per model

**Wskaźnik Kappa Cohena — wszystkie modele**

![](./cohenkappapermodel.png)

Najwyższą kappę uzyskał DistilBERT (`0.5609`), bardzo blisko ALBERT i RoBERTa, natomiast modele GPT mają niską zgodność ponad przypadek (`~0.18-0.19`) - prawie 3 razy lepsza zgodność modeli BERT.

### 6.6 Ewaluacja hierarchiczna (grupy sentymentu)

**Ewaluacja hierarchiczna — grupy sentymentu**

![](./hierarchiczna.png)

Po zredukowaniu złożoności problemu do wymiaru sentymentu (pogrupowaniu 28 emocji w 4 grupy sentymentu) modele BERT wykazują lepsze wyniki niż w klasyfikacji 28-klasowej, jednak sam wzrost jest wyraźnie wyższy w przypadku modeli GPT (około dwukrotny).

### 6.7 Najczęściej mylone pary emocji

Ranking pokazujący, które emocje model najczęściej myli - stanowi to dobre uzupełnienie interpretacji macierzy pomyłek. Pomaga zinterpretować, czy błędy jakie model popełnia są sensowne (emocje bliskie) lub przypadkowe.

**Najczęściej mylone pary — ALBERT**

![](./mylone-pary-albert.png)

**Najczęściej mylone pary — DistilBERT**

![](./mylone-pary-distilbert.png)

**Najczęściej mylone pary — RoBERTa**

![](./mylone-pary-roberta.png)

**Najczęściej mylone pary — Gemma 2**

![](./mylone-pary-gemma2.png)

**Najczęściej mylone pary — Llama 3**

![](./mylone-pary-llama3.png)

Dla modeli BERT praktycznie wszystkie mylone pary emocji uwzględniają etykietę neutral - modele uczą się "strzelać" w najbardziej popularną klasę. W przypadku modeli GPT również mamy głównie do czynienia z pomyłkami z klasą neutral, jednak mamy także kilka sensownych par pomyłek, jak gratitude-joy (gemma2) czy gratitude-approval (llama3).

### 6.8 Zgodność między modelami

Została zmierzona zgodność między modelami na podstawie współczynnika Kappa odpowiedzi modeli - pokazuje to czy modele działają podobnie. 

**Zgodność między modelami (Kappa)**

![](./zgodnosc-miedzymodelami.png)

Widać, że modele BERT mają między sobą większą zgodność niż modele GPT. Modele GPT i BERT między sobą mają zgodność bardzo niską - oznacza to że mylą się i odpowiadają dobrze w zupełnie innych przypadkach.

### 6.9 Test McNemara

**Test McNemara — pary modeli**

![](./test-mcnemara.png)

Różnice BERT vs GPT są istotne statystycznie na wspólnym podzbiorze testowym.

### 6.10 Jakość vs latencja

**Jakość (Macro F1) vs latencja**

![](./jakosc-vs-latencja.png)

DistilBERT daje najlepszy kompromis jakość/szybkość, a GPT są o rzędy wielkości wolniejsze.

---

## 7. Wnioski końcowe

Przeprowadzony eksperyment pozwolił na szczegółową analizę problemu rozpoznawania emocji w tekstach oraz na bezpośrednie porównanie różnych architektur sztucznych sieci neuronowych. Na podstawie uzyskanych wyników i analizy błędów można wysnuć następujące wnioski:

Różnice w skuteczności między modelami BERT a GPT okazały się istotne statystycznie (test McNemara, p < 0.05) na wspólnym podzbiorze testowym, co potwierdza, że przewaga fine-tuningu nie jest przypadkowa. Jednocześnie modele BERT wykazują wysoką wzajemną zgodność, podczas gdy zgodność BERT–GPT jest niska — obie rodziny popełniają błędy i trafiają w zupełnie innych przypadkach.
Ewaluacja hierarchiczna pokazała, że po zredukowaniu problemu do 4 grup sentymentu modele GPT poprawiają wynik niemal dwukrotnie — znacznie bardziej niż modele BERT. Oznacza to, że GPT w trybie zero-shot poprawnie wychwytują ogólny wydźwięk wypowiedzi, a ich główną słabością jest rozróżnianie 28 subtelnych, bliskoznacznych emocji.

Projekt pokazał, że w zadaniu wieloklasowej klasyfikacji emocji, mniejsze modele enkoderowe (rodzina BERT) poddane procesowi douczania z nadzorem radzą sobie znacznie lepiej niż modele generatywne (LLM) działające w trybie zero-shot. Wytrenowany model RoBERTa osiągnął ponad dwukrotnie wyższy wynik Macro-F1 niż modele Llama 3 czy Gemma 2. Oznacza to, że transfer wiedzy na konkretną domenę i strukturę etykiet jest kluczowy dla precyzyjnego rozpoznawania cech tekstu.

Zastosowanie modeli GPT o dużej liczbie parametrów pomimo uzycia kwantyzacji 4-bitowej w zadaniach masowej klasyfikacji tekstów jest nieefektywne obliczeniowo. Model DistilBERT, będący najlżejszą zastosowaną architekturą, osiągnął bardzo dobre wyniki klasyfikacyjne, będąc przy tym tysiące razy szybszym w inferencji od modeli GPT.

Najlepszy wynik jakościowy uzyskał model RoBERTa (najwyższy wskaźnik Macro F1).
Czas poświęcony na fine-tuning modeli BERT i na ich testowanie na pełnym zbiorze danych był niższy niż czas testowania dwóch modeli GPT jedynie na 1000 próbek ze zbioru testowego. Oznacza to, że modele GPT żeby osiągnąć podobną skuteczność w klasyfikacji tekstu potrzebowałyby zużycia dużo większej ilości czasu i zasobów obliczeniowych - są to jednak modele znacznie cięższe. Do zadań klasyfikacji tekstu opłacalniejszym wyborem jest fine-tuning modeli BERT - poradziły sobie one z tym zadaniem całkiem nieźle, a wynik mógłby być jeszcze lepszy, gdybyśmy przeprowadzili trening na większym zbiorze danych, z wykorzystaniem większej ilości epok.

Największym zidentyfikowanym wyzwaniem w projekcie był silny brak balansu w zbiorze danych, ze szczególnym uwzględnieniem dominacji klasy neutral - ponad 35% zbioru treningowego. Spowodowało to zjawisko wybierania przez modele najbezpieczniejszej predykcji. Gdy sieć napotykała tekst z subtelnie zarysowaną emocją, wolała sklasyfikować go jako neutralny, aby zminimalizować błąd.

Mimo trudności z precyzyjnym wskazaniem jednej z 28 bardzo podobnych emocji, badane modele z rodziny BERT wykazały bardzo dobre zrozumienie szerszego kontekstu wypowiedzi. Wskazuje na to wysoka trafność Top-3 (ok. 87%) oraz wyniki przy mapowaniu klas na 4 podstawowe wskaźniki sentymentu. Sieci te rzadko mylą emocje o skrajnie różnej polaryzacji, a ich pomyłki zachodzą głównie w obrębie bliskich klastrów semantycznych.

