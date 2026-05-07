# Dokumentacja procesu

Ten plik dokumentuje **jak** pracowałem/am nad mini-projektem — jakie narzędzia AI wykorzystałem, jakie prompty pisałem, jakie decyzje podjąłem i co nie zadziałało.

> **PROCESS.md jest tak samo ważny jak kod.** Prowadzący ocenia świadome korzystanie z narzędzi AI — to jest kurs o aspektach AI.

---

## Narzędzia AI

| Narzędzie                                                 | Do czego używałem                                                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ChatGPT](https://chat.openai.com?utm_source=chatgpt.com) | Generowanie i adaptacja kodu (PointNet, PointLIME, PointSHAP), debugowanie błędów implementacyjnych oraz pomoc w przygotowaniu dokumentacji projektowej |
| [Claude](https://claude.ai?utm_source=chatgpt.com)        | Generowanie fragmentów kodu pomocniczego, analiza błędów związanych z wymiarami tensorów oraz wsparcie przy przygotowaniu skryptów wizualizacyjnych     |
| [ChatGPT](https://chat.openai.com?utm_source=chatgpt.com) | Korekta i poprawa tekstów w README, dopracowanie opisów metod, wniosków oraz dokumentacji tak, aby były spójne językowo i poprawne merytorycznie        |


## Prompty

> Nie wklejaj outputu z AI — tylko prompty, które wpisywałeś/aś.

### [Kategoria 1, np. "Generowanie kodu"]

1. Przykład 1- generowanie datasetu i dataloadera
```

I need a data pipeline for training a classification model on 3D Gaussian point cloud .ply files.
- The model input should contain point positions together with Gaussian features like scale, rotation, and opacity. 
- Every sample should have a fixed number of points so batches are consistent during training. 
- The dataset should be organized into class folders, where each folder represents one class (for example:
dataset/chair/*.ply, dataset/table/*.ply, dataset/lamp/*.ply) and labels should be assigned automatically based on folder names.
- The pipeline should support batching point clouds even when their original sizes are different, while keeping track of which points are valid and which are padding.

```

**Kontekst:** 

- Pisanie dataloderów i datasetów jest schematycznym procesem, który może być zautomatyzowany poprzez narzędzia AI. Dodatkowo wymaga on odczytywania plików z dysku i dokładne łączenie i processing ścieżek. Podając przykład struktury mojego folderu wiem że wygenerowany kod będzie spójny, i nie będzie ryzyka literówek. Dodatkowo analiza kodu w obiektach Dataset/ Dataloder jest prosta co pozwala mi szybko zweryfikować czy logika zaproponowana przez model jest poprawna czy nie.

2. Przykład 2- dostosowanie kodu PointLime pod moje potrzeby
```
Please adapt the PointLIME code sent below, to my project setup and data pipeline. I need it to work with my existing PointNet model implementation and load the model from an external checkpoint instead of using an internal model definition. The script should accept a .ply point cloud file as input through command-line arguments and generate an explanation output as a .ply file with point-wise importance heatmap. Keep the original LIME logic unchanged as much as possible, only adjust the model interface, input handling, and output saving.
```
**Kontekst:** 
- Kod który miałem bazował z gotowego repozytorium i był zapisany w jednym skrypcie. Chciałem wykorzystać logikę z kodu autorów, ale żeby dostosować jedynie format wejścia i wyjścia, który będzie mi pasował. Wiedziałem, że nie wpłynie to na kluczową implementację metody, a jedynie na operacje związane z wejściem i wyjściem danych do modelu.

3. Przykład 3- Dodanie funkcjonalności zapisywania .ply z kolorami odpowadającymi ważności z SHAPA"
```
Please modify the SHAP explanation script so that the computed Shapley values are saved as a .ply point cloud heatmap, where point colors represent feature importance (warmer colors = higher importance). Keep the original point positions and map SHAP values directly to the point colors for visualization.
```

**Kontekst:** 
- Prosta operacja wyjścia, która wymaga jedynie zapisania koloru, tak aby obiekt był łatwy do interpretacji. Bardzo łatwe do sprawdzenia, wizualizując obiekt.

### [Kategoria 2- Operacje nie związane z generowaniem kodu]

1. Przykład 1- Zmiana formatu pliku z .csv na markdown
```
- Convert the results provided below in CSV format into a Markdown (.md) table. Keep the same columns and preserve the original structure of the data.
```

**Kontekst:** 
- Prosta operacja do zmiany "formatu" danych, aby wyniki były łatwiejsze w prezentacji.

2. Przykład 2- Wyjaśnienie różnicy między metodami
```
Please explain how PointSHAP and PointLIME work in the context of 3D point cloud classification and how they can be adapted from general explainability methods to point-based data. Describe the main differences between them in terms of methodology (local surrogate modeling vs Shapley-value attribution), computational cost, and interpretation of point importance. Also explain how their outputs can be visualized as point-wise heatmaps on .ply files and what aspects should be compared visually (e.g., concentration of important regions, continuity of highlighted areas, consistency with object semantics) to evaluate the quality of the explanations.
```

3. Przykład- poprawianie błędów w tekście
```
Popraw błędy językowe, stylistyczne i interpunkcyjne w poniższym tekście, zachowując jego sens, strukturę oraz techniczny charakter, ale sprawiając, żeby brzmiał naturalnie i profesjonalnie.
```
## Decyzje

1. **Wybór PointNet jako głównej architektury**
   Wybrano PointNet jako główny model, ponieważ osiąga dobrą skuteczność klasyfikacji przy zachowaniu prostoty architektury oraz szybkiego czasu inferencji. Alternatywy takie jak PointNet++ czy PointNeXt dawały nieco lepsze wyniki, jednak były bardziej złożone i mniej praktyczne w analizie explainability.

2. **Dobór parametrów treningu i explainability**
   Parametry treningu oraz metod wyjaśnialności zostały dobrane na podstawie wartości stosowanych w istniejących publikacjach oraz oficjalnych implementacjach. Głównym celem było zachowanie kompromisu pomiędzy jakością wyników a czasem wykonywania eksperymentów, szczególnie istotnym w metodach post-hoc.

3. **Wizualizacja wyjaśnialności jako heatmapy**
   Wyniki explainability zostały przedstawione jako heatmapy zapisane bezpośrednio na chmurach punktów w formacie `.ply`, co umożliwia intuicyjną analizę przestrzenną istotności punktów. Wartości zostały przeskalowane do wspólnej skali kolorów, dzięki czemu możliwe jest łatwe porównywanie wyników między metodami oraz różnymi obiektami.


## Co nie zadziałało

1. **Za mało czasu**
   Pierwotnie plan zakładał analizę explainability dla kilku modeli działających na obiektach 3D (PointNet, PointMLP, PointNet++, PointNeXt) oraz porównanie ich zachowania na tych samych przykładach poprawnych i błędnych klasyfikacji. Ze względu na bardzo wysoki koszt czasowy generowania wyjaśnień oraz dłuższy czas treningu bardziej złożonych modeli, zakres projektu został ograniczony do jednego modelu — PointNet.

2. **Duża złożoność obliczeniowa problemu**
   Zarówno LIME, jak i SHAP działają iteracyjnie, wielokrotnie modyfikując dane wejściowe i uruchamiając inferencję modelu, co powoduje bardzo długi czas generowania pojedynczego wyjaśnienia. Problem częściowo rozwiązano poprzez agresywne próbkowanie punktów obiektu, co znacząco skróciło czas obliczeń kosztem pewnej utraty szczegółowości.

3. **Problemy z doborem parametrów explainability**
   Dobór liczby perturbacji, liczby próbek oraz poziomu segmentacji obiektu miał duży wpływ na jakość i stabilność generowanych wyjaśnień. Zbyt mała liczba iteracji prowadziła do niestabilnych i mało czytelnych heatmap, natomiast większa liczba poprawiała jakość kosztem znacząco dłuższego czasu wykonania. Po paru iteracjach udało się znaleźć optymalny zestaw parametrów, który działał na większosci przypadków, natomiast zdarzały się obiekty dla których metoda zwracała jednokolorowe heatmapy"

## Iteracje

1. **v1**
   W pierwszej wersji projektu założeniem było przygotowanie 3–4 modeli do klasyfikacji obiektów 3D, które nie posiadają natywnie wbudowanych mechanizmów wyjaśnialności, a następnie zastosowanie do nich metod LIME i SHAP. Celem było porównanie zarówno skuteczności modeli, jak i jakości generowanych wyjaśnień — szczególnie w przypadkach zgodnych i rozbieżnych predykcji.

2. **v2**
   Pierwotny zakres projektu okazał się zbyt szeroki względem dostępnych zasobów czasowych i obliczeniowych, szczególnie ze względu na wysoki koszt generowania wyjaśnień metodami post-hoc. W związku z tym zakres został zawężony do jednego modelu — PointNet — przy jednoczesnym zachowaniu wyników benchmarku klasyfikacyjnego dla pozostałych modeli.

3. **v3**
   Finalna wersja projektu skupiła się na szczegółowej analizie wizualizacji explainability dla wybranych przypadków poprawnych i błędnych klasyfikacji oraz na ich jakościowym porównaniu. Obecna struktura projektu została przygotowana tak, aby w przyszłości możliwe było łatwe rozszerzenie analizy o kolejne zaimplementowane modele.
