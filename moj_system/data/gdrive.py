# -*- coding: utf-8 -*-
import os
import io
import logging
import tempfile
import pandas as pd
import socket
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload

class GDriveClient:
    def __init__(self, credentials_path=None):
        # Automatyczne pobieranie głównego folderu projektu ze zmiennej środowiskowej
        self.root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        
        if credentials_path is None:
            self.credentials_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        else:
            self.credentials_path = credentials_path
            
        self.service = self._get_service()

    def _get_service(self):
        if not os.path.exists(self.credentials_path):
            logging.warning(f"Brak pliku credentials w: {self.credentials_path}")
            return None
            
        try:
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            socket.setdefaulttimeout(60) 
            return build("drive", "v3", credentials=creds, cache_discovery=False)
        except Exception as e:
            logging.error(f"Bląd inicjalizacji serwisu Drive: {e}")
            return None

    def find_file_id(self, parent_id: str, filename: str) -> str:
        if not self.service: return None
        try:
            parent_query = f"'{parent_id}' in parents" if parent_id else "'root' in parents"
            query = f"name='{filename}' and {parent_query} and trashed=false"
            
            # DODAJ num_retries=5
            results = self.service.files().list(q=query, fields="files(id,name)").execute(num_retries=5)
            files = results.get("files", [])
            return files[0]["id"] if files else None
        except (ConnectionResetError, socket.timeout):
            logging.warning("Połączenie z GDrive zerwane. Odświeżam serwis...")
            self.service = self._get_service()
            return self.find_file_id(parent_id, filename) # Ponowna próba

    def download_csv(self, folder_id: str, filename: str, sep=",", encoding="utf-8") -> pd.DataFrame:
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

    def upload_file(self, folder_id: str, local_path: str, filename: str = None):
        """Universal uploader for CSV, TXT, PNG and JSON files."""
        if not self.service: return None
        if not filename:
            filename = os.path.basename(local_path)
            
        # Automatyczne wykrywanie typu pliku
        mimetype = "text/csv"
        if filename.endswith(".png"): mimetype = "image/png"
        elif filename.endswith(".json"): mimetype = "application/json"
        elif filename.endswith(".txt"): mimetype = "text/plain"
            
        existing_id = self.find_file_id(folder_id, filename)
        media = MediaFileUpload(local_path, mimetype=mimetype, resumable=True)
        
        if existing_id:
            self.service.files().update(fileId=existing_id, media_body=media).execute()
            logging.info(f"Zaktualizowano plik na Drive: {filename}")
            return existing_id
        else:
            metadata = {"name": filename, "parents": [folder_id]}
            result = self.service.files().create(body=metadata, media_body=media, fields="id").execute()
            logging.info(f"Utworzono nowy plik na Drive: {filename}")
            return result["id"]

    # Dla kompatybilności wstecznej z resztą skryptów (np. data_updater):
    def upload_csv(self, folder_id: str, local_path: str, filename: str = None):
        return self.upload_file(folder_id, local_path, filename)