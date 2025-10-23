# Dane Przykładowe

## Avocado Dataset

Plik `avocado.csv` zawiera dane o sprzedaży awokado w USA.

### Kolumny:

- **Date**: Data obserwacji
- **AveragePrice**: Średnia cena awokado
- **Total Volume**: Całkowita objętość sprzedaży
- **4046, 4225, 4770**: Sprzedaż według kodów PLU
- **Total Bags**: Liczba toreb
- **Small/Large/XLarge Bags**: Rozmiary toreb
- **type**: Typ awokado (conventional/organic)
- **year**: Rok
- **region**: Region sprzedaży

### Przykładowe zastosowania:

1. **Regresja**: Przewidywanie AveragePrice
2. **Klasyfikacja**: Przewidywanie type (conventional vs organic)
3. **Analiza czasowa**: Trendy w czasie

### Źródło:

Hass Avocado Board
https://www.kaggle.com/datasets/neuromusic/avocado-prices

## Wymagania Formatów

### CSV
- Separator: automatycznie wykrywany (`,`, `;`, `\t`, `|`)
- Encoding: UTF-8 (zalecane), latin-1, iso-8859-1
- Pierwsza linia: nagłówki kolumn

### Excel
- Format: .xlsx lub .xls
- Arkusz: pierwszy (domyślnie)
- Nagłówki: pierwsza linia

### Parquet
- Format Apache Parquet
- Kompresja: dowolna wspierana

### JSON
- Format: records lub table
- Struktura: lista obiektów lub obiekt z danymi

## Dobre Praktyki

1. **Nazwy kolumn**: Bez spacji, znaków specjalnych (automatycznie sanityzowane)
2. **Braki danych**: Obsługiwane (imputacja automatyczna)
3. **Typy danych**: Automatycznie wykrywane i konwertowane
4. **Rozmiar**: Zalecane <100MB dla optymalnej wydajności