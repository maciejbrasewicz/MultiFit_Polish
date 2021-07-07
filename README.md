# MultiFit_Polish
MultiFit: pre-trained language model on polish Wikipedia

# Transfer learning

Ludzie mają nieodłączną zdolność przekazywania wiedzy między zadaniami. To, co zdobywamy jako wiedzę podczas wykonywania jednego zadania, wykorzystujemy do rozwiązywania innych zadań. Im bardziej powiązane zadania, tym łatwiej nam przenieść lub wykorzystać naszą wiedzę. 

Do niedawna algorytmy uczenia maszynowego i głębokiego uczenia były zaprojektowane do pracy w izolacji – trenowane do rozwiązywania określonych zadań. Transfer learning to pomysł przezwyciężenia odizolowanego paradygmatu uczenia maszynowego i wykorzystania wiedzy zdobytej podczas wykonywania jednego zadania do rozwiązania zadań podobnych. W skrócie, uczenie maszynowe wykorzystujące transfer learning odnosi się do paradygmatu uczenia maszynowego, w którym algorytm wyodrębnia wiedzę z jednego lub więcej scenariuszy zastosowania, aby pomóc zwiększyć efektywność uczenia się w scenariuszu docelowym.

Można powiedzieć, że transfer learning to zbiór technik i metod, które zajmują się sposobami, w jakich systemy uczenia maszynowego mogą przystosowywać się do nowych sytuacji, nowych zadań oraz nowych środowisk. Na transfer learning możemy spojrzeć jak na ogólną koncepcję wysokiego poziomu, która rozwinęła się na wczesnym etapie uczenia maszynowego, modelowania statystycznego oraz psychologii edukacji. Chociaż niniejsza praca dotyczy uczenia transferowego w kontekście uczenia maszynowego, to jednak wysokopoziomowa koncepcja transfer learningu jest podobna także w innych dziedzinach nauki. Koncepcja transferowego uczenia maszynowego przeszła przez dziesiątki lat ewolucji - idea ta narodziła się już we wczesnych latach badań nad  AI, ponieważ naukowcy uznawali zdolność do przekazywania wiedzy za jeden z fundamentalnych aspektów inteligencji. Umiejętność uczenia się na podstawie małej ilości danych wydaje się być szczególnie silnym aspektem ludzkiej inteligencji; adaptacja jest wrodzoną zdolnością istot inteligentnych, zatem sztucznie inteligentni agenci z pewnością powinni być także obdarzeni tą zdolnością.

Kluczową idea metodyki uczenia transferowego głosi, że jeśli wytrenujemy model uczenia maszynowego w jednej domenie/zadaniu, możemy wykorzystać ten „wstępnie wytrenowany” model w podobnej domenie lub zadaniu osiągając lepsze wyniki niż w przypadku wykorzystania tego modelu bez wstępnego trenowania.

Transfer learning badany był pod kątem różnych terminologii stosowanych w sztucznej inteligencji, takich jak ponowne wykorzystanie wiedzy (ang. knowledge reuse), uczenie się przez analogię, adaptacja domeny (ang. domain adaptation), szkolenie wstępne (ang. pre-training) , dostrajanie (ang. fine-tuning), i tak dalej. W dziedzinie psychologii edukacji, transfer umiejętności ma podobne pojęcie jak transfer learning w uczeniu maszynowym: „przenoszenie umiejętności, przyzwyczajeń i odruchów nabytych w ramach nauki jednej z dziedzin na inną dziedzinę”. Transfer uczenia się w dziedzinie edukacji ma także taki sam cel, jak transfer learning w uczeniu maszynowym - oba dotyczą procesu uczenia się w jednym kontekście i zastosowania zdobytej wiedzy w innym kontekście – oba próbują odpowiedzieć na pytanie jak skutecznie się uczyć? Transfer Learning, mimo kompletnie różnych dziedzin i terminologii, w duchu jest podobny: wykorzystać wcześniejsze doświadczenia, aby poprawić skuteczność pewnych działań w przyszłości.

# Motywacja

Istnieje wiele powdów, aby rozwijać nowatorskie techniki transfer learningu. Dane są jak ogień – trudno przewidzieć, co będzie stanowiło źródło motywacji do rozwoju transfer learningu w przyszłości, jednak już w obecnym czasie nie brakuje powodów do rozwoju tych technik:

Prywatność użytkowników i bezpieczeństwo danych to bardzo ważne kwestie –  w wielu przypadkach trzeba współpracować z innymi organizacjami, wykorzystując wiele zestawów danych. Zestawy danych mają różnych właścicieli i nie można ich ujawniać ze względów prywatności lub bezpieczeństwa. Temat prywatności w data science to problem istotny i obszerny. Zachęcam do zapoznania się z „The private AI series” (Andrew Trask).

Brak wysokiej jakości danych: tradycyjne metody uczenia maszynowego często nie dają się dobrze uogólniać na nowe scenariusze, co może skutkować przeuczeniem modeli (ang. overfitting). Uczenie maszynowe wykorzystujące metody transfer learningu jest rozwiązaniem, które może sprostać tym wyzwaniom, gdyż dzięki niemu możemy wykorzystać wiele danych pomocniczych i modeli zewnętrznych, aby następnie dostosować je w celu rozwiązywania docelowych problemów.

Personalizacja to kolejny istotny problem – niezwykle ważne jest oferowanie spersonalizowanej usługi każdemu klientowi/użytkownikowi według indywidualnych upodobań i żądań. Jednak w wielu rzeczywistych przypadkach możemy zebrać tylko bardzo niewiele danych osobowych od pojedynczego użytkownika. w tym przypadku tradycyjne metody uczenia maszynowego nie spiszą się dobrze, gdyż próbujemy dostosować ogólny model do konkretnej sytuacji - tradycyjne modele ML mają problem z czymś, co określa się jako problem zimnego startu (ang. cold start problem). Na przykład, danych oznaczonych dotyczących oceny użytkownika w systemie rekomendacji może być zbyt mało, aby umożliwić stworzenie wysokiej jakości systemu rekomendacji. Załóżmy, że chcemy zbudować  system rekomendacji książek w nowej aplikacji do zakupów online. Powiedzmy, że domena książki jest tak nowa, że nie zarejestrowaliśmy wielu jej transakcji. Jeśli zastosujemy nadzorowaną metodologię uczenia maszynowego w budowaniu prognozy, można przypuszczać, że model ten nie osiągnie wiarygodnych predykcji. Jednak stosując transfer learning, można poszukać pomocy w pokrewnej, dobrze rozwiniętej domenie, takiej jak domena rekomendacji filmów. Dzięki wykorzystaniu uczenia transferowego możemy znaleźć podstawowe podobieństwa między dwoma zestawami danych – możemy wydobyć „esencję” z tych zestawów danych. Uczenie maszynowe wykorzystujące transfer learning może pomóc w promowaniu sztucznej inteligencji w mniej rozwiniętych obszarach zastosowań, a także w mniej rozwiniętych technicznie obszarach geograficznych  – na przykład, kraju, który nie posiada dużych zbiorów danych tekstowych dotyczących rekomendacji w danej domenie.

Wreszcie, kiedy my, ludzie, uczymy się jakiejś nowej umiejętności, to w pewnym momencie, aby przejść na wyższy poziom potrzebujemy nowego rodzaju danych – potrzebujemy informacji z  nietypowych, rzadkich sytuacji.  W niektórych obszarach uczenia maszynowego dane z przypadków granicznych często są niezwykle trudne do uzyskania, np.: w przypadku diagnostyki medycznej, leków, czy chociażby w przypadku autonomicznych samochodów (niewystarczająca ilość danych z niebezpiecznych sytuacji). Ta uwaga na temat korzyści ze stosowania metod transfer earningu dotyczy technologii symulacyjnej. W niektórych dziedzinach, takich jak robotyka czy projektowanie leków, angażowanie się w eksperymenty w prawdziwym świecie jest zbyt kosztowne i niepewne. W robotyce, aby autonomiczny pojazd mógł bezpiecznie się poruszać, musi posiadać dane treningowe obejmujące jak największa ilość sytuacji, które mogą się wydarzyć na prawdziwej drodze. Na przykład, samochód autonomiczny może brać udział w wypadku na niezliczoną ilość sposobów; jednak powodowanie wypadków samochodowych w prawdziwym świecie po to, aby wyciągnąć z nich informacje (dane treningowe) jest zbyt drogie.


Podsumowując, uzyskiwanie i oznaczanie nowych danych jest po prostu kosztowne i wymaga dużego wysiłku, co stanowi główną przeszkodę w zastosowaniu sztucznej inteligencji w prawdziwym świecie. W świecie, który zalewa powódź nieistotnych danych, przejrzystość informacji to potęga. Ilość danych  rośnie w zastraszającym tempie. Jednak w obszarach, w których danych brakuje, według wszelkiego prawdopodobieństwa, w przyszłości wciąż będzie ich brakować. 

Jeśli nie nauczymy się pozyskiwać odpowiednich danych do naszych modeli oraz nie opanujemy nowych metod, dzięki którym  będziemy mogli wykorzystywać dane uzyskane w jednej dziedzinie, aby rozwiązać problem – nie zdołamy wdrożyć uczenia maszynowego w zupełnie nowych dziedzinach życia. 
 
 Uczenie maszynowe wykorzystujące transfer learning jest jednak czymś więcej niż tylko drobnym ulepszeniem. Transfer learning zdaniem wielu naukowców i badaczy danych może znacznie przyśpieszyć postęp w kierunku AGI. Andrew Ng w 2016 roku w swoim wystąpieniu „Nuts and bolts of building AI applications using Deep Learning” powiedział, że po nadzorowanym uczeniu – Transfer Learning będzie kolejnym motorem sukcesu komercyjnego uczenia maszynowego.
 
Możliwość transferu wiedzy z jednego zadania do drugiego pozwala algorytmom uczenia maszynowego rozszerzyć zakres ich zastosowania poza ich oryginalne stworzenie. Ta zdolność uogólnienia pomaga uczynić AI bardziej dostępną i niezawodną w wielu obszarach, w których zasoby, takie jak moc obliczeniowa, dane czy bezpieczeństwo mogą niewystarczające. W pewnym sensie nauka transferu pozwala promować uczenie maszynowe jako technologie bardziej dostępną – technologię, która może służyć każdemu.

# Proces uczenia transferowego

![image](https://user-images.githubusercontent.com/49028274/124733933-f3597c00-df14-11eb-9fd4-2cc8dde4f90b.png)

W oparciu o definicję Pan i Yang 2014 możemy sformułować różne sposoby kategoryzacji istniejących badań nad uczeniem się w ramach transferu. Na przykład, w oparciu o jednorodność przestrzeni cech i / lub przestrzeni etykiet, możemy podzielić uczenie transferu na dwa ustawienia: (1) uczenie homogeniczne i (2) uczenie heterogeniczne, których definicje wychodzą poza readme na githubie (wzory).

Podczas projektowania algorytmu uczenia maszynowego wykorzystującego Transfer Learning, należy wziąć pod uwagę następujące trzy główne problemy badawcze: (1) Kiedy przenieść,  (2) Co przenieść (3) Jak przenieść.

**Kiedy przenosić**, próbuje odpowiedzieć na pytanie, w jakich sytuacjach należy dokonać transfer learningu. Podobnie, chcielibyśmy wiedzieć, w jakich sytuacjach wiedza nie powinna być przenoszona (transferowana). W niektórych sytuacjach, gdy domena źródłowa i docelowa nie są ze sobą powiązane, transfer learning może się oczywiście nie powieść. Większość aktualnych badań na temat transfer learningu  skupia się na odpowiedziach na pytania: „co przenieść” i „jak przenieść”. 1 Jednak, jak uniknać transferu ujemnego, jest ważną otwartą kwestią, która przyciąga coraz więcej uwagi.

**Co przenieść**, decyduje o tym, którą część wiedzy można przekazać do nowej domeny lub zadania. Część wiedzy jest specyficzna dla poszczególnych domen lub zadań, a pewna wiedza może być dla nich wspólna. Zwróć uwagę, że termin „wiedza” jest bardzo ogólny i różni się on w zależności od danego kontekstu.

**Jak przenieść**, określa metodę, jaką wykorzystaliśmy stosując transfer learning. Odpowiedzi na to pytanie kategoryzują algorytmy uczenia przez transfer:

1) **Algorytmy oparte na instancjach** (ang. instance-based algorithms), do których przekazywana  wiedza odpowiada wagą przypisanych do instancji źródłowych;
2) **Algorytmy oparte na cechach** (ang. feature-based algorithms), w których przekazywana wiedza odpowiada podprzestrzeni liniowej łączonej przez cechy w domenie źródłowej i docelowej;
3) **Algorytmy oparte na modelach** (model-based algorithms), w których przekazywana wiedza wbudowana jest w część modeli z domeny źródłowej;
4) **Algorytmy oparte na relacjach** (ang. relation-based algorithms), w których przekazywana wiedza opowiada się za regułami określającymi relacje między jednostkami w domenie źródłowej.

Każdy z tych rodzajów uczenia transferowego odpowiada za sposób, w jaki wiedza może być pomyślnie  przekazana i wykorzystana w domenie docelowej. W przypadku podejścia opartego na instancjach, dane oznaczone w domenie źródłowej nie mogą być wykorzystane bezpośrednio ze względu na różnicę domen; część z nich można wykorzystać ponowne dla domeny docelowej po ponownym ważeniu lub ponownym próbkowaniu. W tym przypadku, instancje o dużej wadze, można uznać za „wiedzę”, która może zostać przeniesiona. Oczywiście, w wielu rzeczywistych aplikacjach tylko część przestrzeni funkcji z domeny źródłowej i domeny docelowej pokrywają się, co oznacza, że wiele funkcji nie może być wykorzystana jako „most” do Transfer Learningu. W rezultacie, korzystanie z metod opartych na instancjach, często nie prowadzi do pozytywnych rezultatów.

Podejście oparte na cechach jest w tym przypadku bardziej obiecujące. Kluczową ideą tego podejścia jest nauczenie się „dobrej” reprezentacji funkcji zarówno dla domeny źródłowej, jak i docelowej, tak aby poprzez rzutowanie danych na nową reprezentację, dane oznaczone w domenie źródłowej mogły być ponownie wykorzystane do szkolenia klasyfikatora dla domeny docelowej. W tym sposobie za „wiedzę”, która ma być przekazywana między domenami można uznać wyuczoną funkcję reprezentacyjną.

Podejścia oparte na modelach zakładają, że modele w domenie źródłowej i domenie docelowej mają wspólne parametry lub hiperparametry. Kluczową ideą w podejściu opartym na modelach jest dobrze wyszkolony model w domenie źródłowej; „przechwytuje” on wiele przydatnych struktur, które są ogólne i można je przenieść do bardziej precyzyjnego modelu docelowego. 

W ostatnim czasie, szeroko stosowanatechnika wstępnego szkolenia (ang. pre-training) w transfer learningu jest podejściem opartym na modelach. W najprostszym wydaniu tego podejścia szkolimy model głębokiego uczenia w domenie źródłowej przy użyciu wystarczającej ilości danych, które mogą być zupełnie inne niż dane w domenie docelowej. Po przeszkoleniu modelu głębokiego uczenia stosuje się niewielką ilość danych oznaczonych z domeny docelowej do precyzyjnej regulacji części parametrów wstępnie wyuczonego modelu, na przykład do dostrojenia parametrów kilku ostatnich warstw sieci neuronowej.

# Taksonomia

Taksonomię uczenia transferowego z rozszerzeniem dla NLP można zobaczyć na poniższym rysunku. Podział ten zaproponowany został przez S. Ruder’a w 2019 roku. Nie sposób omówić krótko całej taksonomii, dlatego też w tej sekcji skupimy się jedynie na niektórych elementach taksonomii Transfer Learningu.

![image](https://user-images.githubusercontent.com/49028274/124735821-b1313a00-df16-11eb-92af-310c8be05c08.png)

**Szkolenie wstępne modeli językowych (ang. pre-training LM)**
Ideą stojącą za wstępnie przeszkolonymi modelami językowymi (pre-training LM) jest stworzenie „czarnej skrzynki”, która uczy się języka z dużego zbioru danych tekstowych (np. Wikipedii), a następnie może zostać poproszona o wykonanie konkretnego zadania NLP w tym języku. Chodzi o to, aby stworzyć maszynową równowartość „dobrze oczytanego człowieka”, którego dzięki temu można szybciej nauczyć nowego, specyficznego zadania.

![image](https://user-images.githubusercontent.com/49028274/124736800-a6c37000-df17-11eb-993f-ea4d702deafa.png)

# Opis zadań

**UNIVERSAL LANGUAGE MODEL FINE-TUNING (ULMFIT)**

Indukcyjne uczenie metodą transferu znacząco wpłynęło na wizję komputerową, ale początkowo nie powiodło się, gdy zostało zastosowane w NLP. Zaproponowana przez S. Ruder Universal Language Model Fine-tuning (ULMFiT) - skuteczna metoda uczenia transferowego, która może być zastosowana do każdego zadania w NLP –  znacznie przewyższyła najnowsze techniki stosowane w sześciu zadaniach klasyfikacji tekstu w języku angielskim, zmniejszając błąd o 18-24% w większości zbiorów danych. Co więcej, przy zaledwie stu oznakowanych przykładach, zaproponowane techniki, odpowiadają wydajności treningu od podstaw na 100x więcej ilości danych oznaczonych.

Jeremy Howard oraz Sebastian Ruder stwierdzili, że problem nie istniał w koncepcji dopracowania modelu językowego (LM), lecz w sposobie podejścia do problemu.1 Ze względu na to, że modele językowe (LM) są znacznie płytsze w porównaniu z modelami widzenia komputerowego (CV), wymagało to innego podejścia. Zaproponowali oni ULMFiT, który wykorzystuje dyskryminacyjne dostrajanie (ang. discriminative fine-tuning) i skośne trójkątne współczynniki uczenia (ang. slanted triangular learning rates - STLR) do nauki cech specyficznych dla zadania. W zaproponowanej metodzie model (klasyfikator) jest dostosowany do zadania docelowego za pomocą stopniowego odmrażania (ang. gradual unfreezing) oraz STLR.

**EFFICIENT MULTI-LINGUAL LANGUAGE MODEL FINE-TUNING (MULTIFIT)**

Metoda MultiFiT została zaproponowana w 2019 roku jako rozszerzenie metody ULMFiT1.  MultiFiT rozszerza ULMFiT, aby uczynić go jeszcze bardziej wydajnym w modelowaniu języków innych niż język angielski. Zaproponowana metoda zamiast sieci AWD-LSTM wykorzystuje sieć QRNN, co sprawia, że jest jeszcze bardziej skuteczna.

Ponadto, tokenizacja zastąpiona została tokenizacją słów podrzędnych. Tokenizacja słów podrzędnych ma jedną bardzo ważną właściwość dla wielojęzycznego modelowania języka, mianowicie tokeny słów podrzędnych skuteczniej odwzorowują fleksje (odmiany wyrazów), dzięki czemu dobrze nadają się do języków bogatych morfologicznie.

Metoda MultiFiT wykorzystuje także szereg innych usprawnień; szybsze szkolenie osiągnięte zostało dzięki zastąpieniu stopniowego odmrażania poprzez zastosowanie cosinusowego wariantu polityki jednego cyklu. 


  
# Przebieg szkolenia wstępnego – zadanie źródłowe

W sztucznej inteligencji stosowanej do NLP możliwe jest trenowanie modeli językowych w wybranym języku do zadania jakim jest przewidywanie następnego słowa w zdaniu. W niniejszym podrozdziale zostanie opisany przebieg szkolenia modelu MultiFiT dla języka polskiego. Chcąc zweryfikować możliwości tego modelu językowego i tym samym przyczynić się do rozwoju polskiej społeczności NLP, wytrenowałem model językowy na korpusie danych z polskiej Wikipedii składającym się ze 100 milionów tokenów. Czas treningu wyniósł 8 godzin i 53 minuty przy użyciu jednego procesora graficznego Tesla V100.

Model językowy MultiFiT nie jest wystarczająco głęboki pod względem liczby warstw i parametrów, aby skorzystać z korpusu szkoleniowego o rozmiarze większym niż 100 milionów tokenów. Jednak korpus w języku polskim pobrany z Wikipedii składał się z około 136 milionów tokenów, zatem pewna liczba artykułów została usunięta, aby sprowadzić go do korpusu liczącego około 100 milionów tokenów. Rysunki poniżej przedstawiają przebieg szkolenia modelu językowego.

![image](https://user-images.githubusercontent.com/49028274/124728349-cf476c00-df0f-11eb-8248-cdd23ae299ce.png)

Gdzie: epoch oznacza cykl treningu; train_loss  oraz valid_loss oznacza funkcję kosztu; error_rate to poziom błędu; accuracy to dokładność naszego modelu; perplexity  traktowana jest jako wskaźnik mówiący o trudności, jaką modelowi sprawia problem predykcji przewidywania następnego słowa, natomiast time oznacza czas trwania jednego cyklu treningu modelu językowego. Następny rysunek przedstawia dokładność w każdym cyklu treningowym oraz czas trwania całego treningu.

![image](https://user-images.githubusercontent.com/49028274/124728619-087fdc00-df10-11eb-8857-0aac24706972.png)

Całkowity koszt wytrenowania modelu językowego zależy od użytej do tego celu instancji maszyny wirtualnej. W moim przypadku została wykorzystana instancja z pojedynczym procesorem graficznym Tesla V100. Koszt wytrenowania modelu z wykorzystaniem wyżej opisanej instancji wyniósł około $75 , zakładając jednak, że nie napotkamy żadnych błędów podczas treningu, koszt ten nie powinien przekroczyć $30.

Wyniki są bardzo zachęcające i przedstawiają się lepiej niż w przypadku języka francuskiego, który w tej samej konfiguracji osiągnął dokładność 43.7%  oraz języka portugalskiego, w którym dokładność wyniosła 39.7%. Warto zaznaczyć, że te wyniki nie muszą oznaczać, że model językowy, który osiągnął większa dokładność w tym zadaniu (zadaniu źródłowym) sprawdzi się lepiej w zadaniu niższego rzędu (zadaniu docelowym).

# Podsumowanie – zadanie docelowe

Chociaż problem stosowania mowy nienawiści w Internecie istnieje tak długo, jak długo istnieje komunikacja za pośrednictwem Internetu między ludźmi, pojawienie się nowych urządzeń, takich jak smartfony i tablety, które pozwalają na korzystanie z tego medium nie tylko w domu, pracy czy szkole, ale także w ruchu, jeszcze bardziej pogłębiło problem. Zwłaszcza ostatnia dekada, podczas której usługi serwisów społecznościowych (SNS), takich jak Facebook i Twitter, szybko zyskały na popularności, ujawniła problem nieetycznych zachowań w środowiskach internetowych. Jest to problem cyberprzemocy (CB), definiowanego jako wykorzystywanie otwartych środków komunikacji online, takich jak fora internetowe, czy SNS, do przekazywania szkodliwych i niepokojących informacji o osobach prywatnych, często dzieciach i studentach. 

Aby poradzić sobie z tym problemem, naukowcy rozpoczęli badania nad problemem cyberprzemocy w celu automatycznego wykrywania wpisów internetowych zawierających szkodliwe informacje. Po wielu latach badań zebrano wystarczającą bazę wiedzy na temat tego problemu dla języków krajów dobrze rozwiniętych, takich jak USA czy Japonia. Niestety, wciąż prawie nic w tej sprawie nie zrobiono dla języka polskiego. W tym zadaniu należy określić, czy zgłoszenie internetowe można sklasyfikować jako część narracji cyberprzemocy, czy też nie. Dane składają się z komentarzy i wpisów zebrane z ogólnie dostępnych dyskusji na Twitterze. 

W tym zadaniu należy rozróżnić pomiędzy normalnymi/nieszkodliwymi tweetami (klasa: 0) oraz tweetami, które zawierają jakiekolwiek szkodliwe informacje (klasa: 1). Obejmuje to cyberprzemoc, mowę nienawiści i związane z nią zjawiska.

Co istotne, wysoka wydajność modelu w zadaniu źródłowym nie wskazuje na to, czy ostateczny model będzie się dobrze sprawdzał w zadaniu docelowym. Zaproponowane podejścia w zadaniu docelowym osiągały wyniki osiągające od 57 do 63% punktów F1 score. W przypadku tego zadania, które należy do problemu klasyfikacji, wyniki są konkurencyjne dla modeli, które do treningu wymagają kilkunastokrotnie większej mocy obliczeniowej. Dla porównania, model PolBert wytrenowany dla źródłowego, jakim jest przewidywanie następnego słowa w zdaniu, przy użyciu około 1860 000000 tokenów w języku polskim, osiąga w tym samym zadaniu 52.7% punktów F1 score zgodnie z pracą Pre-training Polish Transformer-Based Language Models at Scale. Oczywiście, nie można powiedzieć, że zaproponowany model jest lepszy; można jedynie powiedzieć, że lepiej sprawdził się w przypadku tego zadania. 

Podsumowując, model został wstępnie wytrenowany na nieustrukturyzowanych danych z polskiej Wikipedii – 100 milionów tokenów, osiągając od **56** do **62** punktów F1 score w oficjalnym zadaniu docelowym z konkursu PolEval 2019, jakim było wykrywanie mowy nienawiści w języku polskim. Zaproponowany model cechował się dużym wahaniem wyników. Pozostaje otwartym pytaniem, czy wysoka wydajność modelu MultiFiT w przypadku zadania źródłowego – przewidywania następnego słowa w zdaniu – przełoży się na wysoką wydajność w przypadku zadań docelowych. Przykładem tego założenia może być wspomniany model PolBert, który znacznie przewyższa zaproponowany model w zadaniu źródłowym oraz wielu zadaniach docelowych, jednak w przypadku zaproponowanego zadania związanego z klasyfikacją osiąga słabsze wyniki. Dlatego, aby dokonać skutecznej ewaluacji modeli NLP, które uwzględniają w swojej metodyce transfer learning, należy przejść przez cały proces transfer learningu uwzględniając szkolenie wstępne do zadania źródłowego i ewaluację, dostrajanie (fine-tuning) do zadania docelowego i ewaluację w zadaniu docelowym. 

Wyniki są obiecujące, lecz niejednoznaczne. Praca warta jest dalszej analizy w przyszłości w celu sprawdzenia, **czy wstępne trenowanie modeli językowych w domenie, której bliżej do zadania docelowego, może wpływać korzystniej na wyniki, niż w przypadku trenowania wstępnego w domenie, która jest bardziej odległa od domeny docelowej.
