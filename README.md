Oto kompletny i czytelny plik README.md dla Twojego nowego systemu. Skopiuj go i zapisz w głównym folderze Twojego projektu.

Quantitative Strategy Framework

System do automatycznej analizy, optymalizacji i weryfikacji strategii inwestycyjnych (Single Asset, Pension Multi-Asset, Global Equity). System został zrefaktoryzowany do architektury modułowej, zapewniającej wysoką spójność danych, powtarzalność wyników (OOS alignment) oraz łatwą skalowalność.

Struktura Projektu

moj_system/
├── config.py           # Centralna konfiguracja: parametry, siatki gridów, rejestr aktywów
├── core/               # Silnik obliczeniowy (strategie, metryki, regresja OLS, robustness)
├── data/               # Zarządzanie danymi (Updater, GDriveClient, Builder, KNF tools)
├── reporting/          # Generowanie logów, raportów i wykresów
└── scripts/            # Punkty wejścia (CLI) dla zadań analitycznych
outputs/                # Wyniki backtestów i logi (git-ignored)

Instalacja

Wymagania systemowe:
System wymaga zainstalowanego silnika Tesseract OCR oraz Poppler (dla przetwarzania PDF).

Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-pol poppler-utils libgl1

Windows: Pobrać binarki Poppler i Tesseract, dodać do PATH lub ustawić zmienne środowiskowe POPPLER_PATH i TESSERACT_CMD.

Biblioteki Python:

pip install -r requirements.txt
Szybki Start (CLI)

System sterowany jest z poziomu katalogu moj_system/scripts/.

1. Codzienne uruchomienie strategii

Użyj uniwersalnego runnera:

# Dla pojedynczego assetu
python moj_system/scripts/daily_runner.py --asset WIG20TR --stop_mode atr

# Dla portfela emerytalnego
python moj_system/scripts/daily_runner.py --asset PENSION

2. Optymalizacja parametrów (Sweep)

Znajdowanie najlepszych ustawień (Train/Test) i typów stop-lossa:
python moj_system/scripts/sweep_optimizer.py --mode PENSION --n_mc 100

3. Weryfikacja odporności (Robustness)

Pełna diagnostyka (Monte Carlo + Bootstrap) dla wybranej konfiguracji:

python moj_system/scripts/validate_robustness.py --mode SINGLE --asset WIG20TR --train 8 --test 2 --stop fixed

4. Aktualizacja funduszy KNF

Pobranie najnowszych notowań z API KNF i dopasowanie ich do tickerów Stooq:

python moj_system/scripts/refresh_knf.py --update
Konfiguracja

Wszystkie kluczowe parametry strategii znajdują się w moj_system/config.py:

ASSET_REGISTRY: Rejestr wszystkich instrumentów, ich źródeł danych (Stooq/Drive) oraz dedykowanych siatek parametrów.

BASE_GRIDS / BOND_GRIDS: Definicje przestrzeni poszukiwań (Grid Search).

Automatyzacja (GitHub Actions)

System posiada w pełni zautomatyzowane workflowy (.github/workflows/):

Daily Runner: Uruchamia strategie codziennie o 00:00 UTC.

OCR Download: Pobiera i przetwarza dane PPE codziennie o 23:00 UTC.

Weekly Fund Review: Odświeża rankingi funduszy TFI w każdy poniedziałek o 04:00 UTC.

Wymagane Zmienne Środowiskowe (Secrets)

Aby system działał w chmurze, należy skonfigurować w GitHub Secrets (lub pliku .env):

GDRIVE_FOLDER_ID

GOOGLE_CREDENTIALS (JSON string)

ZIP_URL, ZIP_PASSWORD, INT_FILE_NAME, FOLDER_NAME (dla OCR)


