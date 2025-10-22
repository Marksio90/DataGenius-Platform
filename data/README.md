# /data – zestawy danych

Tu trzymamy dane wejściowe do demonstracji i testów E2E.

## Pliki

- **avocado.csv** – mały, syntetyczny wycinek (kilkanaście rekordów) na podstawie popularnego zbioru „Avocado Prices”.
  - Przydatny do: wczytania przykładu w aplikacji, EDA, szybkiego treningu modeli.
  - Kolumny:
    - `date` (datetime, ISO) – data obserwacji
    - `average_price` (float) – średnia cena
    - `total_volume` (float) – wolumen sprzedaży
    - `type` (string: `conventional`/`organic`)
    - `year` (int)
    - `region` (string)

## Wskazówki dot. danych własnych

- Obsługiwane formaty: **CSV / XLSX / Parquet / JSON**.
- Rekomendacje:
  - Kodowanie: UTF-8
  - Separator CSV: `,` (kropka jako separator dziesiętny)
  - Nagłówki w pierwszym wierszu
- Nazwy kolumn są automatycznie normalizowane (snake_case, unikalne).
- Duże pliki: aplikacja defensywnie próbkowuje do raportów EDA.

## Jak wykorzystywane w aplikacji

- `DataService.load_example()` zwraca `avocado.csv`.
- Fingerprint danych (hash) służy do cache’owania raportów i artefaktów.

> Uwaga: plik jest syntetyczny (do demonstracji), **nie** zawiera oryginalnych danych z pełnego zbioru Kaggle.
