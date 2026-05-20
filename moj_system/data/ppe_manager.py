# -*- coding: utf-8 -*-
"""
moj_system/data/ppe_manager.py
==============================
Moduł łączący długoterminową historię funduszy PPE z codziennymi
odczytami OCR (filtered_table.csv). Zwraca zunifikowany DataFrame
oraz automatycznie uaktualnia pełną bazę (full_ppe_history.csv) na Google Drive.
"""

import logging

import pandas as pd

from moj_system.config import DATA_DIR
from moj_system.data.gdrive import GDriveClient

# Mapowanie zunifikowanych nazw kolumn
PPE_COLUMN_MAPPING = {
    "equity": "equity",
    "bond":   "bond",
    "mmf":    "mmf",
}

def build_continuous_ppe_data(
    folder_id:        str,
    credentials_path: str,
) -> pd.DataFrame | None:
    """
    Pobiera historię oraz najnowsze dane OCR z GDrive, łączy je,
    usuwa duplikaty, ujednolica kolumny i wgrywa zaktualizowaną
    historię (full_ppe_history.csv) z powrotem na GDrive.
    """
    client = GDriveClient(
        credentials_path=credentials_path
    )
    
    if not client.service:
        logging.error(
            msg="PPE Manager: No GDrive service available."
        )
        return None

    # 1. Pobranie statycznej historii (sep=",")
    hist_df = client.download_csv(
        folder_id=folder_id, 
        filename="full_ppe_history.csv", 
        sep=","
    )
    
    # 2. Pobranie najnowszych danych z OCR (sep=";")
    ocr_df = client.download_csv(
        folder_id=folder_id, 
        filename="filtered_table.csv", 
        sep=";"
    )

    if hist_df is None or hist_df.empty:
        logging.error(
            msg="PPE Manager: Missing or empty full_ppe_history.csv on Drive."
        )
        return None

    # Oryginalne nagłówki do późniejszego zapisu na GDrive
    original_header = [
        "data wyceny",
        "BEZPIECZNA JESIEŃ SFIO SUBFUNDUSZ AKCJI",
        "BEZPIECZNA JESIEŃ SFIO SUBFUNDUSZ ZRÓWNOWAŻONY",
        "BEZPIECZNA JESIEŃ SFIO SUBFUNDUSZ STABILNEGO WZROSTU",
        "BEZPIECZNA JESIEŃ SFIO SUBFUNDUSZ OBLIGACJI",
        "BEZPIECZNA JESIEŃ SFIO SUBFUNDUSZ KONSERWATYWNY"
    ]

    # --- 3. Standaryzacja pliku z historią ---
    hist_df.columns = ["Date", "equity", "balanced", "stable", "bond", "mmf"]
    
    hist_df["Data"] = pd.to_datetime(
        arg=hist_df["Date"], 
        dayfirst=True, 
        errors="coerce"
    )
    hist_df.dropna(
        subset=["Data"], 
        inplace=True
    )
    hist_df.set_index(
        keys="Data", 
        inplace=True
    )
    hist_df.drop(
        columns=["Date"], 
        inplace=True
    )
    hist_df.sort_index(
        inplace=True
    )

    # --- 4. Standaryzacja pliku z OCR ---
    if ocr_df is not None and not ocr_df.empty:
        # Ponieważ OCR zapisywano bez header=False, Pandas wczytał pierwszy wiersz jako nazwy kolumn.
        # Odzyskujemy go i nadajemy poprawne nazwy kolumn (takie same jak dla historii).
        if len(ocr_df.columns) == 6:
            first_row = pd.DataFrame(
                data=[ocr_df.columns.tolist()]
            )
            ocr_df.columns = range(6)
            ocr_df = pd.concat(
                objs=[first_row, ocr_df], 
                ignore_index=True
            )
            ocr_df.columns = ["Date", "equity", "balanced", "stable", "bond", "mmf"]
        
        ocr_df["Data"] = pd.to_datetime(
            arg=ocr_df["Date"], 
            dayfirst=True, 
            errors="coerce"
        )
        ocr_df.dropna(
            subset=["Data"], 
            inplace=True
        )
        ocr_df.set_index(
            keys="Data", 
            inplace=True
        )
        ocr_df.drop(
            columns=["Date"], 
            inplace=True
        )
        ocr_df.sort_index(
            inplace=True
        )

        # Odrzucamy z historii te daty, które mamy już z OCR
        hist_df = hist_df.loc[hist_df.index < ocr_df.index.min()]
        combined_df = pd.concat(
            objs=[hist_df, ocr_df], 
            axis=0
        )
    else:
        combined_df = hist_df

    # --- 5. Konwersja WSZYSTKICH kolumn na numeryczne ---
    # Konwertujemy wszystkie 5 funduszy, by wyeksportowany plik CSV był w 100% poprawny
    all_fund_cols = ["equity", "balanced", "stable", "bond", "mmf"]
    for col in all_fund_cols:
        if combined_df[col].dtype == object:
            combined_df[col] = combined_df[col].astype(str).str.replace(
                pat=",", 
                repl=".", 
                regex=False
            )
        combined_df[col] = pd.to_numeric(
            arg=combined_df[col], 
            errors="coerce"
        )

    combined_df.ffill(
        inplace=True
    )
    
    # --- 6. Upload uaktualnionej historii na GDrive ---
    if ocr_df is not None and not ocr_df.empty:
        history_export = combined_df.copy().reset_index()
        history_export.columns = original_header
        
        # Zapis daty w formacie oryginalnym: dd/mm/yyyy
        history_export["data wyceny"] = history_export["data wyceny"].dt.strftime(
            date_format="%d/%m/%Y"
        )
        
        hist_out_path = DATA_DIR / "full_ppe_history.csv"
        hist_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_export.to_csv(
            path_or_buf=hist_out_path, 
            index=False, 
            sep=","
        )
        
        logging.info(
            msg="Uploading updated full_ppe_history.csv to Google Drive..."
        )
        client.upload_csv(
            folder_id=folder_id,
            local_path=str(hist_out_path),
            filename="full_ppe_history.csv"
        )

    # --- 7. Zapis lokalny zunifikowanego pliku (dla cache/debugowania) ---
    cache_path = DATA_DIR / "ppe_combined.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(
        path_or_buf=cache_path, 
        sep=";"
    )

    logging.info(
        msg=f"PPE Data built successfully. Rows: {len(combined_df)}, latest date: {combined_df.index.max().date()}"
    )

    return combined_df
