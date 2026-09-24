# -*- coding: utf-8 -*-
import io
import logging
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


class GDriveClient:
    def __init__(
        self, 
        credentials_path: str | None = None,
    ) -> None:

        self.root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")

        if os.name == 'nt' and os.environ.get("GOOGLE_CREDENTIALS"):
            self.credentials_data = os.environ.get("GOOGLE_CREDENTIALS")
            self.credentials_path = None
        else:
            self.credentials_path = credentials_path or os.path.join(tempfile.gettempdir(), "credentials.json")
            self.credentials_data = None

        self.service = self._get_service()

    def _get_service(self) -> object | None:
        import json
        import socket
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        try:
            if self.credentials_data:
                creds_info = json.loads(self.credentials_data)
                creds = service_account.Credentials.from_service_account_info(
                    creds_info,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            elif self.credentials_path and Path(self.credentials_path).exists():
                creds = service_account.Credentials.from_service_account_file(
                    filename=self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
            else:
                logging.warning(msg=f"Brak poprawnych danych uwierzytelniających.")
            return None

            socket.setdefaulttimeout(120)
            return build(serviceName="drive", version="v3", credentials=creds, cache_discovery=False)
        except Exception as e:
            logging.error(msg=f"Bląd inicjalizacji serwisu Drive: {e}")
            return None

    def find_file_id(
        self, 
        parent_id: str, 
        filename:  str,
    ) -> str | None:

        if not self.service:
            return None
        try:
            parent_query = f"'{parent_id}' in parents" if parent_id else "'root' in parents"
            query = f"name='{filename}' and {parent_query} and trashed=false"

            # DODAJ num_retries=5
            results = (
                self.service.files().list(q=query, fields="files(id,name)").execute(num_retries=5)
            )
            files = results.get("files", [])
            return files[0]["id"] if files else None
        except (ConnectionResetError, socket.timeout):
            logging.warning("Połączenie z GDrive zerwane. Odświeżam serwis...")
            self.service = self._get_service()
            return self.find_file_id(parent_id, filename)  # Ponowna próba

    def download_csv(
        self, 
        folder_id: str, 
        filename:  str, 
        sep:       str = ",", 
        encoding:  str = "utf-8",
    ) -> pd.DataFrame | None:

        file_id = self.find_file_id(folder_id, filename)
        if not file_id:
            return None

        buf = io.BytesIO()
        request = self.service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        buf.seek(0)
        try:
            return pd.read_csv(buf, sep=sep, encoding=encoding, engine="python")
        except Exception as e:
            logging.error(f"Bląd dekodowania CSV: {e}")
            return None

    def upload_file(
        self, 
        folder_id:  str, 
        local_path: str, 
        filename:   str | None = None,
    ) -> str | None:
        from googleapiclient.http import MediaFileUpload
        import time

        service: Any = self.service
        if not service:
            return None
            
        if not filename:
            filename = Path(local_path).name

        mimetype = "text/csv"
        if filename.endswith(".png"):
            mimetype = "image/png"
        elif filename.endswith(".json"):
            mimetype = "application/json"
        elif filename.endswith(".txt"):
            mimetype = "text/plain"

        existing_id = self.find_file_id(parent_id=folder_id, filename=filename)
        
        # POPRAWKA: Pancerna pętla uploadu z mechanizmem Retry
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Wymuszamy nowe otwarcie pliku przy każdej próbie, bo MediaFileUpload
                # podczas błędu może zostawić kursor na końcu pliku.
                media = MediaFileUpload(filename=local_path, mimetype=mimetype, resumable=True)
                
                if existing_id:
                    service.files().update(
                        fileId=existing_id, 
                        media_body=media
                    ).execute(num_retries=5) # Wbudowany mechanizm ponawiania pakietów
                    
                    logging.info(msg=f"Zaktualizowano plik na Drive: {filename}")
                    return existing_id
                else:
                    metadata = {"name": filename, "parents": [folder_id]}
                    result = service.files().create(
                        body=metadata, 
                        media_body=media, 
                        fields="id"
                    ).execute(num_retries=5)
                    
                    logging.info(msg=f"Utworzono nowy plik na Drive: {filename}")
                    return result["id"]
                    
            except Exception as e:
                logging.warning(msg=f"Upload attempt {attempt + 1}/{max_attempts} failed for {filename}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)  # Odczekanie przed ponowieniem
                    
                    # Czasami błąd wynika z wygasłego tokenu lub zerwanego gniazda.
                    # Twardy reset połączenia z serwerami Google:
                    self.service = self._get_service()
                    service = self.service
                else:
                    logging.error(msg=f"All upload attempts failed for {filename}")
                    return None
                    
        return None

    # Dla kompatybilności wstecznej z resztą skryptów (np. data_updater):
    def upload_csv(
        self, 
        folder_id:  str, 
        local_path: str, 
        filename:   str | None = None,
    ) -> str | None:
    
        return self.upload_file(folder_id, local_path, filename)
