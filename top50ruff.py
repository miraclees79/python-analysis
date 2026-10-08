# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:35:29 2026

@author: adamg
"""

import subprocess
import sys

python_exe = sys.executable
# Uruchamiamy ruff i przechwytujemy wyjście
process = subprocess.Popen(
    [python_exe, "-m", "ruff", "check", "."],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
)

# Wypisujemy tylko pierwsze 50 linii
count = 0
print("\n" + "="*80)
print("TOP 500 lines RUFF ERRORS")
print("="*80)

for line in process.stdout:
    print(line.strip())
    count += 1
    if count >= 500:
        process.terminate() # Zatrzymujemy proces po 500 liniach
        break
