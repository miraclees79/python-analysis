# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 22:11:49 2026

@author: adamg
"""

import os
import io
import logging
import tempfile
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload

class GDriveClient:
    def __init__(self, credentials_path=None):
        if credentials_path is None:
            self.credentials_path = os.path.join(tempfile.gettempdir(), "credentials.json")
        else:
            self.credentials_path = credentials_path
            
        self.service = self._get_service()

    def _get_service(self):
        if not os.path.exists(self.credentials_path):
            logging.warning(f"Brak pliku credentials w: {self.credentials_path}. Tryb offline.")
            return None
            
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def find_file_id(self, folder_id: str, filename: str) -> str:
        if not self.service: return None
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id,name)").execute()
        files = results.get("files", [])
        return files[0]["id"] if files else None

    def download_csv(self, folder_id: str, filename: str, sep=",", encoding="utf-8") -> pd.DataFrame:
        # Zmieniliśmy domyślny sep z ";" na ","
        file_id = self.find_file_id(folder_id, filename)
        if not file_id:
            logging.warning(f"Nie znaleziono pliku na Drive: {filename}")
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
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, sep=sep, encoding="cp1250", engine="python")

    def upload_csv(self, folder_id: str, local_path: str, filename: str = None):
        if not self.service: return None
        if not filename:
            filename = os.path.basename(local_path)
            
        existing_id = self.find_file_id(folder_id, filename)
        media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
        
        if existing_id:
            self.service.files().update(fileId=existing_id, media_body=media).execute()
            logging.info(f"Zaktualizowano plik na Drive: {filename}")
            return existing_id
        else:
            metadata = {"name": filename, "parents": [folder_id]}
            result = self.service.files().create(body=metadata, media_body=media, fields="id").execute()
            logging.info(f"Utworzono nowy plik na Drive: {filename}")
            return result["id"]