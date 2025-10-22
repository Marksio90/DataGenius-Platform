# TMIV – UX GUIDE (Skeleton)

Przewodnik projektowo-implementacyjny dla UI TMIV Advanced ML Platform. Opisuje nawigację, przepływy użytkownika, wzorce komponentów, stany, treści i dostępność. Dokument jest „żywy” – aktualizujemy przy zmianach w UI.

---

## 1) Cel i zakres

- **Cel**: umożliwić analitykom i zespołom biznesowym wykonanie pełnego cyklu „od danych do rekomendacji” bez kodu.
- **Zakres**: Streamlit UI (desktop-first, responsywność podstawowa), widoki, komponenty, stany i copy.

---

## 2) Persony i scenariusze

**P1: Analityk danych (primary)**
- Potrzeby: szybki import, eksploracja, baseline modeli, porównania, eksport wyników.
- Sukces: ma gotowy PDF/ZIP i intuicyjne rekomendacje.

**P2: Manager biznesowy**
- Potrzeby: zrozumiałe wykresy, główne metryki, „co dalej?”.
- Sukces: widzi klarowne wnioski i ryzyka.

**P3: Inżynier ML / MLOps**
- Potrzeby: artefakty, powtarzalność, monitoring, DB snapshot.
- Sukces: integruje TMIV z pipeline’ami.

---

## 3) Architektura informacji (IA)

**Nawigacja główna (sidebar):**
1. 📊 Analiza Danych  
2. 🤖 Trening Modelu  
3. 📈 Wyniki i Wizualizacje  
4. 💡 Rekomendacje  
5. 📚 Dokumentacja

**Wzorce:**
- Każda sekcja = jedna „misja” użytkownika.
- Status globalny (czy są dane? czy jest wynik?) w sidebarze.
- Bez zagnieżdżanych expandersów w narzędziach admin/monitoringu.

---

## 4) Kluczowe przepływy (flows)

### 4.1 Import i EDA
- **Wejście**: plik CSV/XLSX/Parquet/JSON lub dataset demo.
- **Kroki**: wczytaj → sanityzacja typów → podsumowanie → korelacje → (opcjonalnie) profil HTML.
- **Wyjście**: gotowy DataFrame w pamięci sesji.

### 4.2 Planowanie i trening
- **Wejście**: DF + wybór kolumny target.
- **Kroki**: budowa planu (heurystyki) → trening kilku modeli → leaderboard.
- **Wyjście**: najlepszy model + metryki + FI.

### 4.3 Wyniki i wizualizacje
- **Wejście**: wynik treningu.
- **Kroki**: przewidywania, wykresy (ROC/PR/CM/kalibracja lub diagnostyka regresji), FI.
- **Wyjście**: zestaw PNG + tabela metryk.

### 4.4 Rekomendacje
- **Wejście**: metryki, FI, kontekst.
- **Kroki**: generacja (LLM/heurystyki) → lista punktów działania.
- **Wyjście**: checklist „co poprawić dalej”.

### 4.5 Eksporty
- **PDF Explainability** i **ZIP artefaktów** z przycisku – pokazujemy ścieżkę pliku i link otwierający.

---

## 5) Szablony ekranów

### 5.1 „📊 Analiza Danych”
- **Header**: tytuł + krótki opis.
- **Sekcje**:
  - Import: uploader + przycisk „Użyj przykładowego”.
  - Podsumowanie (metryki „Rows/Cols/Missing%”).
  - Typy i braki (JSON/mini tabela).
  - Korelacje (heatmap tabela).
  - Profil EDA (przycisk → generuj → link do HTML).
- **Pusty stan**: call-to-action „wczytaj plik lub użyj demo”.

### 5.2 „🤖 Trening Modelu”
- **Formularz**: target, strategia, test_size, random_state, [Trenuj].
- **Po treningu**: Leaderboard (tabela), FI (tabela), highlight best model.

### 5.3 „📈 Wyniki i Wizualizacje”
- **Klasyfikacja**: ROC, PR, CM, Calibration.
- **Regresja**: error dist, pred_vs_true, residuals_vs_pred.
- **FI (plot)**: bar chart top-K.
- **Eksporty**: [PDF] [ZIP] – pokaż ścieżkę i link.

### 5.4 „💡 Rekomendacje”
- Lista punktów, krótkie i konkretne zdania; język korzyści.

### 5.5 „📚 Dokumentacja”
- Kroki pracy, FAQ, wskazówki dot. formatów danych.

---

## 6) Komponenty i wzorce UI

- **Form**: etykiety po lewej, walidacja inline, wartości domyślne (bez „pustych”).
- **Tabele**: sticky header, sortowanie metryki głównej, max 100 wierszy (paginacja).
- **Wykresy**: jeden wykres = jedna figura (PNG), podpis + opis.
- **Przyciski**: czasowniki (np. „Trenuj modele”, „Zbuduj raport”).
- **Powiadomienia**: `st.success/warning/error/info` z krótką treścią.

---

## 7) Stany i błędy

**Stany:**
- *Empty*: instrukcja „co teraz”.
- *Loading*: spinner + opis akcji (np. „Trenuję…”).
- *Success*: co zostało wytworzone + gdzie to jest.
- *Error*: zwięzły komunikat, bez stacktrace; link „spróbuj ponownie”.

**Zasady copy błędów:**
- Nie obwiniaj użytkownika.
- Nigdy nie pokazuj sekretów/ścieżek systemowych.
- Przykład: „Nie udało się wygenerować profilu EDA. Upewnij się, że pakiet `ydata-profiling` jest zainstalowany lub spróbuj mniejszej próbki.”

---

## 8) Dostępność (A11y)

- Kontrast ≥ 4.5:1 (motyw dark + jasne akcenty).
- Focus states dla elementów sterujących.
- Alternatywny tekst dla obrazów (caption = opis).
- Rozsądne role i nagłówki (H1/H2/H3).
- Klawiatura: przechodzenie po kluczowych kontrolkach formularza.

Checklist:
- [ ] Każdy obraz ma podpis.  
- [ ] Kolory nie są jedynym nośnikiem informacji.  
- [ ] Komunikaty błędów są czytane przez screenreadery (widoczne w DOM).

---

## 9) Treści i mikrocopy

- Zwięźle, po polsku, czasowniki w trybie rozkazującym.  
- Zasada „co? dlaczego? co dalej?” w opisach sekcji.
- Terminy ML w formie zrozumiałej (bez skrótów w nagłówkach, pełna nazwa + skrót w nawiasie w treści).

---

## 10) Wydajność i skalowanie UX

- Profil EDA: budowany na próbce + cache.
- Wykresy: generuj on demand; unikaj wielokrotnego renderowania tych samych obrazów.
- Limity:
  - max 200 kolumn na profil,
  - wykres FI: top 50 (UI: domyślnie top 20),
  - tabele: paginacja po 100.

---

## 11) Wskazówki dla testów UX (QA)

- **Smoke**: import → EDA → trening → jeden wykres → PDF/ZIP.
- **Edge cases**:
  - brak kolumn numerycznych,
  - skrajnie niezbalansowane klasy,
  - bardzo mała próbka (≤ 50 wierszy).
- **Perf**: duży CSV (≥ 100 MB) – UI powinien pozostać responsywny (spinner, komunikaty).

---

## 12) Telemetria UI (opcjonalnie)

Zdarzenia (nazwy przykładowe):
- `ui.load_example_clicked`
- `ui.training_started` / `ui.training_finished`
- `ui.report_built` / `ui.export_zip`
- Atrybuty: `rows`, `cols`, `target`, `problem_type`.

---

## 13) Bezpieczeństwo i prywatność

- Nie logujemy danych wrażliwych.
- Ścieżki do artefaktów pokazujemy użytkownikowi, ale nie wstawiamy treści plików.
- Klucze LLM tylko lokalnie (`secrets.toml` / ENV); brak ekspozycji w UI.

---

## 14) Lokalizacja i formaty

- Język podstawowy: **PL** (krótkie, jasne komunikaty).
- Format liczb: kropka dziesiętna, separator tysięcy nieużywany w metrykach.
- Daty: ISO (YYYY-MM-DD) + czas lokalny przy logach, gdy ma sens.

---

## 15) Style i wygląd

- **Motyw**: dark base + akcent fioletowy (spójnie z `.streamlit/config.toml`).
- **Typografia**: bez udziwnień, semibold w nagłówkach.
- **Odstępy**: wystarczające – „oddychające” panele; unikamy ścisku.

---

## 16) Roadmap UX (pomysły)

- Dashboard startowy (ostatnie runy, skróty).
- Tryb „Presenter” – tylko metryki i kluczowe wykresy na jednym ekranie.
- Progi decyzyjne (threshold slider) z natychmiastową rewizualizacją F1/precision/recall.

---

## 17) Checklista przed MR/Release

- [ ] Pusty stan dla każdej strony.  
- [ ] Komunikaty błędów bez stacktrace.  
- [ ] Dostępność: podpisy obrazów i kontrast.  
- [ ] Brak blokujących spinnerów > 5 s bez opisu akcji.  
- [ ] Linki do artefaktów (PDF/ZIP) działają.  
- [ ] Teksty zwięzłe i ujednolicone (terminologia ML).

---
