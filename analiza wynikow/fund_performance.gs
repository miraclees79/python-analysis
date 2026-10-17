// FundPerformance.gs
//
// Google Apps Script for the Fund Performance workbook.
// Imports fund_ranking_{date}.csv from Google Drive, writes it to the
// "Raw Data" sheet, then rebuilds summary sheets for quick analysis.
//
// Sheet structure this script manages:
//   Raw Data        — full import of latest fund_ranking CSV (all windows)
//   Scorecards 3Y   — 3Y window only, sorted by IR within category
//   Category Summary — one row per category: median IR, % funds IR>0, etc.
//   TFI Summary      — one row per TFI: fund count, median IR, % IR>0
//   Window Comparison — one row per fund, IR/alpha side-by-side across windows
//
// Setup:
//   1. Set GDRIVE_FOLDER_ID in Script Properties
//      (Extensions > Apps Script > Project Settings > Script Properties)
//   2. Run setupTrigger() once to register the daily refresh.
//   3. Run importFundPerformance() manually to seed the sheet on first run.
//
// Trigger: daily, after the GitHub Actions workflow typically completes.
//
// New relative fields include active_return, appraisal_ratio, peer_ir, and benchmark provenance.
// Optional new columns let this importer continue to accept older ranking files.

// ---------------------------------------------------------------------------
// Configuration — set GDRIVE_FOLDER_ID in Script Properties
// ---------------------------------------------------------------------------

function getFolderId() {
  const id = PropertiesService.getScriptProperties().getProperty("GDRIVE_FOLDER_ID");
  if (!id) throw new Error("Script Property GDRIVE_FOLDER_ID is not set.");
  return id;
}

// ---------------------------------------------------------------------------
// Drive helper: find latest fund_ranking_YYYYMMDD.csv
// ---------------------------------------------------------------------------

function findLatestRankingFile(folderId) {
  const folder = DriveApp.getFolderById(folderId);

  // List all files matching the prefix, pick the lexicographically latest
  // (YYYYMMDD suffix sorts correctly as a string)
  const iter    = folder.getFiles();
  let   best    = null;
  let   bestName = "";

  while (iter.hasNext()) {
    const f    = iter.next();
    const name = f.getName();
    if (/^fund_ranking_\d{8}\.csv$/.test(name) && name > bestName) {
      best     = f;
      bestName = name;
    }
  }

  if (!best) throw new Error("No fund_ranking_YYYYMMDD.csv found in Drive folder.");
  Logger.log("Using ranking file: " + bestName);
  return best;
}

// ---------------------------------------------------------------------------
// CSV parse (semicolon-separated, handles quoted fields)
// ---------------------------------------------------------------------------

function parseCsv(text) {
  // Proper RFC-4180-style parser for semicolon-separated CSV.
  //
  // The naive split("\n") approach breaks when a quoted field contains a
  // literal newline — e.g. benchmark descriptions from KNF kid data such as:
  //   "WIBOR6M ustalony 2 dni robocze przed ostatnim dniem\no 30 punktów bazowych."
  // This parser reads character-by-character, respects quoting, and replaces
  // any newline inside a quoted field with a space (cleaner for Sheets).
  //
  // Also pads / truncates every row to the header column count so that
  // setValues() never throws a dimension mismatch.

  text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();

  const rows        = [];
  let   currentRow  = [];
  let   currentField= [];
  let   inQuotes    = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (inQuotes) {
      if (ch === '"') {
        // Escaped quote ("") → literal quote character
        if (i + 1 < text.length && text[i + 1] === '"') {
          currentField.push('"');
          i++;
        } else {
          inQuotes = false;
        }
      } else if (ch === "\n") {
        // Newline inside a quoted field: replace with space to keep
        // the field on a single Sheets row
        currentField.push(" ");
      } else {
        currentField.push(ch);
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ";") {
        currentRow.push(currentField.join("").trim());
        currentField = [];
      } else if (ch === "\n") {
        currentRow.push(currentField.join("").trim());
        currentField = [];
        rows.push(currentRow);
        currentRow = [];
      } else {
        currentField.push(ch);
      }
    }
  }
  // Flush the final field and row
  currentRow.push(currentField.join("").trim());
  if (currentRow.some(f => f !== "")) rows.push(currentRow);

  if (rows.length === 0) return [];

  const ncols = rows[0].length;
  Logger.log("CSV parsed: " + rows.length + " rows, " + ncols + " columns in header.");

  // Normalise every row to header width
  return rows.map((row, i) => {
    if (row.length === ncols) return row;
    if (row.length > ncols) {
      Logger.log("Row " + i + ": " + row.length + " cols — truncating to " + ncols);
      return row.slice(0, ncols);
    }
    const padded = row.slice();
    while (padded.length < ncols) padded.push("");
    return padded;
  });
}

// ---------------------------------------------------------------------------
// Sheet helpers
// ---------------------------------------------------------------------------

function getOrCreateSheet(ss, name) {
  let sheet = ss.getSheetByName(name);
  if (!sheet) {
    sheet = ss.insertSheet(name);
  }
  return sheet;
}

function clearSheet(sheet) {
  sheet.clearContents();
  sheet.clearFormats();
}

// ---------------------------------------------------------------------------
// Column index lookup (0-based from header row)
// ---------------------------------------------------------------------------

function colIndex(headers, name) {
  const idx = headers.indexOf(name);
  if (idx === -1) throw new Error("Column not found in CSV: " + name);
  return idx;
}

function optionalColIndex(headers, name) {
  return headers.indexOf(name);
}

function isRankableIr(record) {
  return record.benchmark_rankable === true &&
    record.benchmark_is_proxy === false &&
    (record.benchmark_quality === "kid_full" || record.benchmark_quality === "kid_partial") &&
    record.benchmark_coverage !== null && record.benchmark_coverage >= 0.8;
}

// ---------------------------------------------------------------------------
// Conditional formatting helpers
// ---------------------------------------------------------------------------

function applyIrFormatting(sheet, dataRange) {
  // Green-yellow-red gradient on IR column
  // dataRange: the full data range to search for the IR column
  const rules = sheet.getConditionalFormatRules();
  const newRules = [];

  // Positive IR → green
  newRules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenNumberGreaterThan(0.5)
    .setBackground("#B7E1CD")
    .setRanges([dataRange])
    .build());

  // Marginally positive → light green
  newRules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenNumberBetween(0.0, 0.5)
    .setBackground("#E8F5E9")
    .setRanges([dataRange])
    .build());

  // Negative IR → light red
  newRules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenNumberLessThan(0.0)
    .setBackground("#FFDEDE")
    .setRanges([dataRange])
    .build());

  sheet.setConditionalFormatRules(newRules.concat(rules));
}

function applyPctIrPositiveFormatting(sheet, range) {
  const rules = sheet.getConditionalFormatRules();
  const newRules = [];

  newRules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenNumberGreaterThanOrEqualTo(0.6)
    .setBackground("#B7E1CD")
    .setRanges([range])
    .build());

  newRules.push(SpreadsheetApp.newConditionalFormatRule()
    .whenNumberLessThan(0.4)
    .setBackground("#FFDEDE")
    .setRanges([range])
    .build());

  sheet.setConditionalFormatRules(newRules.concat(rules));
}

// ---------------------------------------------------------------------------
// MAIN FUNCTION: importFundPerformance()
// ---------------------------------------------------------------------------

function importFundPerformance() {
  const ss       = SpreadsheetApp.getActiveSpreadsheet();
  const folderId = getFolderId();
  const file     = findLatestRankingFile(folderId);
  const text     = file.getBlob().getDataAsString("UTF-8");
  const rows     = parseCsv(text);

  if (rows.length < 2) {
    Logger.log("CSV appears empty — aborting.");
    return;
  }

  const headers  = rows[0];
  const dataRows = rows.slice(1);

  // Validate that every data row has been padded to header width.
  // Log a warning for any that differ — helps diagnose malformed CSV files.
  const expectedCols = headers.length;
  Logger.log("CSV header width: " + expectedCols + " columns, " + dataRows.length + " data rows.");
  dataRows.forEach((row, i) => {
    if (row.length !== expectedCols) {
      Logger.log("WARNING: row " + (i+1) + " has " + row.length +
                 " cols (expected " + expectedCols + ") — padding/truncating.");
      while (row.length < expectedCols) row.push("");
      if (row.length > expectedCols) row.splice(expectedCols);
    }
  });

  // Parse numeric columns we'll need downstream
  const CI = {
    subfundId:    colIndex(headers, "subfundId"),
    stooq_id:     colIndex(headers, "stooq_id"),
    knf_name:     colIndex(headers, "knf_name"),
    tfi_name:     colIndex(headers, "tfi_name"),
    category:     colIndex(headers, "category"),
    classification: colIndex(headers, "classification"),
    riskLevel:    colIndex(headers, "riskLevel"),
    fee:          colIndex(headers, "managementAndOtherFeesRate"),
    entryFee:     colIndex(headers, "maxEntryFeeRate"),
    benchmark:    colIndex(headers, "benchmark"),
    window:       colIndex(headers, "window"),
    ann_return:   colIndex(headers, "ann_return"),
    ann_vol:      colIndex(headers, "ann_vol"),
    sharpe:       colIndex(headers, "sharpe"),
    sortino:      colIndex(headers, "sortino"),
    max_drawdown: colIndex(headers, "max_drawdown"),
    calmar:       colIndex(headers, "calmar"),
    data_years:   colIndex(headers, "data_years"),
    fetch_ok:     colIndex(headers, "fetch_ok"),
    alpha:        colIndex(headers, "alpha"),
    beta:         colIndex(headers, "beta"),
    tracking_error: colIndex(headers, "tracking_error"),
    ir:           colIndex(headers, "ir"),
    active_return: optionalColIndex(headers, "active_return"),
    appraisal_ratio: optionalColIndex(headers, "appraisal_ratio"),
    residual_tracking_error: optionalColIndex(headers, "residual_tracking_error"),
    correlation: optionalColIndex(headers, "correlation"),
    r_squared: optionalColIndex(headers, "r_squared"),
    n_relative_observations: optionalColIndex(headers, "n_relative_observations"),
    peer_ir: optionalColIndex(headers, "peer_ir"),
    pct_ir_pos:   colIndex(headers, "pct_ir_positive"),
    n_ir_windows: colIndex(headers, "n_ir_windows"),
    cat_funds:    colIndex(headers, "category_funds"),
    cat_peers_ok: colIndex(headers, "category_peers_ok"),
    benchmark_source: optionalColIndex(headers, "benchmark_source"),
    benchmark_series_source: optionalColIndex(headers, "benchmark_series_source"),
    benchmark_quality: optionalColIndex(headers, "benchmark_quality"),
    benchmark_frequency: optionalColIndex(headers, "benchmark_frequency"),
    benchmark_coverage: optionalColIndex(headers, "benchmark_coverage"),
    benchmark_components: optionalColIndex(headers, "benchmark_components"),
    benchmark_unresolved: optionalColIndex(headers, "benchmark_unresolved"),
    benchmark_reason: optionalColIndex(headers, "benchmark_reason"),
    benchmark_is_proxy: optionalColIndex(headers, "benchmark_is_proxy"),
    benchmark_rankable: optionalColIndex(headers, "benchmark_rankable"),
    rank_ir:      colIndex(headers, "rank_ir"),
    rank_appraisal_ratio: optionalColIndex(headers, "rank_appraisal_ratio"),
    rank_sharpe:  colIndex(headers, "rank_sharpe"),
    rank_sortino: colIndex(headers, "rank_sortino"),
    rank_calmar:  colIndex(headers, "rank_calmar"),
    rank_return:  colIndex(headers, "rank_return"),
  };

  const fileName = file.getName();
  const runDate  = fileName.replace("fund_ranking_", "").replace(".csv", "");

  // ── Tab 1: Raw Data ──────────────────────────────────────────────────────
  const rawSheet = getOrCreateSheet(ss, "Raw Data");
  clearSheet(rawSheet);
  rawSheet.appendRow(["Imported from: " + fileName + "  |  " + new Date().toISOString()]);
  rawSheet.appendRow(headers);
  if (dataRows.length > 0) {
    rawSheet.getRange(3, 1, dataRows.length, headers.length).setValues(dataRows);
  }
  rawSheet.getRange(2, 1, 1, headers.length)
    .setFontWeight("bold")
    .setBackground("#E8EAF6");
  rawSheet.setFrozenRows(2);
  Logger.log("Raw Data: " + dataRows.length + " rows written.");

  // ── Build parsed data structures ─────────────────────────────────────────
  // Convert to objects for easier manipulation
  function getOptional(row, index) { return index < 0 ? "" : row[index]; }
  function parseNum(v) {
    if (v === null || v === undefined || /^(?:\s*|nan|none|null)$/i.test(String(v))) return null;
    const parsed = Number(v);
    return Number.isFinite(parsed) ? parsed : null;
  }
  function parseBool(v, fallback) {
    if (v === true || v === "True" || v === "true") return true;
    if (v === false || v === "False" || v === "false") return false;
    return fallback === true;
  }
  const records = dataRows.map(r => ({
    subfundId:    r[CI.subfundId],
    stooq_id:     r[CI.stooq_id],
    knf_name:     r[CI.knf_name],
    tfi_name:     r[CI.tfi_name],
    category:     r[CI.category],
    classification: r[CI.classification],
    riskLevel:    parseNum(r[CI.riskLevel]),
    fee:          parseNum(r[CI.fee]),
    entryFee:     parseNum(r[CI.entryFee]),
    benchmark:    r[CI.benchmark],
    window:       r[CI.window],
    ann_return:   parseNum(r[CI.ann_return]),
    ann_vol:      parseNum(r[CI.ann_vol]),
    sharpe:       parseNum(r[CI.sharpe]),
    sortino:      parseNum(r[CI.sortino]),
    max_drawdown: parseNum(r[CI.max_drawdown]),
    calmar:       parseNum(r[CI.calmar]),
    data_years:   parseNum(r[CI.data_years]),
    fetch_ok:     parseBool(r[CI.fetch_ok]),
    alpha:        parseNum(r[CI.alpha]),
    beta:         parseNum(r[CI.beta]),
    tracking_error: parseNum(r[CI.tracking_error]),
    ir:           parseNum(r[CI.ir]),
    active_return: parseNum(getOptional(r, CI.active_return)),
    appraisal_ratio: parseNum(getOptional(r, CI.appraisal_ratio)),
    residual_tracking_error: parseNum(getOptional(r, CI.residual_tracking_error)),
    correlation: parseNum(getOptional(r, CI.correlation)),
    r_squared: parseNum(getOptional(r, CI.r_squared)),
    n_relative_observations: parseNum(getOptional(r, CI.n_relative_observations)),
    peer_ir: parseNum(getOptional(r, CI.peer_ir)),
    pct_ir_pos:   parseNum(r[CI.pct_ir_pos]),
    n_ir_windows: parseNum(r[CI.n_ir_windows]),
    cat_funds:    parseNum(r[CI.cat_funds]),
    cat_peers_ok: parseBool(r[CI.cat_peers_ok]),
    benchmark_source: getOptional(r, CI.benchmark_source) || "legacy",
    benchmark_series_source: getOptional(r, CI.benchmark_series_source),
    benchmark_quality: getOptional(r, CI.benchmark_quality) || "legacy",
    benchmark_frequency: getOptional(r, CI.benchmark_frequency),
    benchmark_coverage: parseNum(getOptional(r, CI.benchmark_coverage)),
    benchmark_components: getOptional(r, CI.benchmark_components),
    benchmark_unresolved: getOptional(r, CI.benchmark_unresolved),
    benchmark_reason: getOptional(r, CI.benchmark_reason),
    benchmark_is_proxy: parseBool(getOptional(r, CI.benchmark_is_proxy), true),
    benchmark_rankable: parseBool(getOptional(r, CI.benchmark_rankable), false),
    rank_ir:      parseNum(r[CI.rank_ir]),
    rank_appraisal_ratio: parseNum(getOptional(r, CI.rank_appraisal_ratio)),
    rank_sharpe:  parseNum(r[CI.rank_sharpe]),
    rank_sortino: parseNum(r[CI.rank_sortino]),
    rank_calmar:  parseNum(r[CI.rank_calmar]),
    rank_return:  parseNum(r[CI.rank_return]),
  }));
  records.forEach(record => {
    if (!isRankableIr(record)) record.rank_ir = null;
  });

  // ── Tab 2: Scorecards 3Y ─────────────────────────────────────────────────
  buildScorecardsSheet(ss, records, "3y", runDate);

  // ── Tab 3: Category Summary ───────────────────────────────────────────────
  buildCategorySummarySheet(ss, records, runDate);

  // ── Tab 4: TFI Summary ────────────────────────────────────────────────────
  buildTfiSummarySheet(ss, records, runDate);

  // ── Tab 5: Window Comparison ──────────────────────────────────────────────
  buildWindowComparisonSheet(ss, records, runDate);

  Logger.log("importFundPerformance complete. Run date: " + runDate);
}


// ---------------------------------------------------------------------------
// Tab 2: Scorecards — one row per fund (primary window), sorted by IR
// ---------------------------------------------------------------------------

function buildScorecardsSheet(ss, records, primaryWindow, runDate) {
  const sheet = getOrCreateSheet(ss, "Scorecards " + primaryWindow.toUpperCase());
  clearSheet(sheet);

  const filtered = records
    .filter(r => r.window === primaryWindow && r.fetch_ok)
    .sort((a, b) => {
      // Sort by category asc, then rank_ir asc within category
      if (a.category < b.category) return -1;
      if (a.category > b.category) return 1;
      return (a.rank_ir || 9999) - (b.rank_ir || 9999);
    });

  const HEADERS = [
    "Category", "Rank (IR)", "Fund Name", "TFI",
    "IR", "Alpha %", "% Windows IR>0", "TE %",
    "Sharpe", "Sortino", "Max DD %", "Ann Return %",
    "Risk Level", "Fee %", "Entry Fee %",
    "Data Years", "Cat. Funds", "Peers OK",
    "Benchmark", "Appraisal Ratio", "Peer IR", "Benchmark Quality",
    "Benchmark Source", "Coverage", "IR Rankable"
  ];

  const titleRow = ["Scorecards — window: " + primaryWindow.toUpperCase() +
                    "  |  Run: " + runDate];
  sheet.appendRow(titleRow);
  sheet.appendRow(HEADERS);

  const dataOut = filtered.map(r => [
    r.category,
    r.rank_ir,
    r.knf_name,
    r.tfi_name,
    r.ir,
    r.alpha !== null ? +(r.alpha * 100).toFixed(2) : null,
    r.pct_ir_pos,
    r.tracking_error !== null ? +(r.tracking_error * 100).toFixed(2) : null,
    r.sharpe,
    r.sortino,
    r.max_drawdown !== null ? +(r.max_drawdown * 100).toFixed(2) : null,
    r.ann_return !== null ? +(r.ann_return * 100).toFixed(2) : null,
    r.riskLevel,
    r.fee !== null ? +(r.fee * 100).toFixed(2) : null,
    r.entryFee !== null ? +(r.entryFee * 100).toFixed(2) : null,
    r.data_years,
    r.cat_funds,
    r.cat_peers_ok ? "✓" : "✗",
    r.benchmark,
    r.appraisal_ratio,
    r.peer_ir,
    r.benchmark_quality,
    r.benchmark_source,
    r.benchmark_coverage,
    r.benchmark_rankable ? "Yes" : "No",
  ]);

  if (dataOut.length > 0) {
    sheet.getRange(3, 1, dataOut.length, HEADERS.length).setValues(dataOut);
  }

  // Formatting
  const headerRange = sheet.getRange(2, 1, 1, HEADERS.length);
  headerRange.setFontWeight("bold").setBackground("#3F51B5").setFontColor("#FFFFFF");
  sheet.setFrozenRows(2);
  sheet.setFrozenColumns(3);

  // IR column = col 5 (E), data rows start at row 3
  if (dataOut.length > 0) {
    const irRange     = sheet.getRange(3, 5, dataOut.length, 1);
    const pctIrRange  = sheet.getRange(3, 7, dataOut.length, 1);
    const maxDdRange  = sheet.getRange(3, 11, dataOut.length, 1);

    applyIrFormatting(sheet, irRange);
    applyPctIrPositiveFormatting(sheet, pctIrRange);

    // Max drawdown: red if worse than -20%
    const ddRules = sheet.getConditionalFormatRules();
    ddRules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenNumberLessThan(-20)
      .setBackground("#FFDEDE")
      .setRanges([maxDdRange])
      .build());
    sheet.setConditionalFormatRules(ddRules);

    // Alternate row shading by category for readability
    let prevCat = "";
    let shade   = false;
    for (let i = 0; i < dataOut.length; i++) {
      const cat = dataOut[i][0];
      if (cat !== prevCat) { shade = !shade; prevCat = cat; }
      if (shade) {
        sheet.getRange(i + 3, 1, 1, HEADERS.length).setBackground("#F5F5F5");
      }
    }
  }

  // Auto-resize key columns
  sheet.autoResizeColumns(1, HEADERS.length);
  Logger.log("Scorecards " + primaryWindow.toUpperCase() + ": " + dataOut.length + " rows.");
}


// ---------------------------------------------------------------------------
// Tab 3: Category Summary
// ---------------------------------------------------------------------------

function buildCategorySummarySheet(ss, records, runDate) {
  const sheet = getOrCreateSheet(ss, "Category Summary");
  clearSheet(sheet);

  const WINDOWS_ORDER = ["1y", "3y", "5y", "all"];

  // Group 3Y records by category for summary stats
  const byCategory = {};
  records.forEach(r => {
    if (!r.fetch_ok) return;
    if (!byCategory[r.category]) byCategory[r.category] = {};
    if (!byCategory[r.category][r.window]) byCategory[r.category][r.window] = [];
    byCategory[r.category][r.window].push(r);
  });

  const HEADERS = [
    "Category",
    // IR statistics include only rankable KID benchmarks.
    "3Y Med IR (KID)", "3Y % IR>0 (KID)", "3Y Med Alpha%", "3Y Med Sharpe", "3Y N Funds",
    "1Y Med IR (KID)", "1Y % IR>0 (KID)",
    "5Y Med IR (KID)", "5Y % IR>0 (KID)",
    "All Med IR (KID)", "All % IR>0 (KID)",
    // Cross-window consistency: funds with IR>0 in ALL windows that have data
  ];

  sheet.appendRow(["Category Summary  |  Run: " + runDate]);
  sheet.appendRow(HEADERS);

  function median(arr) {
    const sorted = arr.filter(v => v !== null).sort((a,b)=>a-b);
    if (!sorted.length) return null;
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
  }
  function pctPositive(arr) {
    const valid = arr.filter(v => v !== null);
    if (!valid.length) return null;
    return +(valid.filter(v=>v>0).length / valid.length).toFixed(3);
  }

  const categories = Object.keys(byCategory).sort();
  const dataOut = categories.map(cat => {
    const wins = byCategory[cat];
    const w3   = wins["3y"] || [];
    const w1   = wins["1y"] || [];
    const w5   = wins["5y"] || [];
    const wall = wins["all"] || [];

    const ir3   = w3.filter(isRankableIr).map(r=>r.ir);
    const ir1   = w1.filter(isRankableIr).map(r=>r.ir);
    const ir5   = w5.filter(isRankableIr).map(r=>r.ir);
    const irAll = wall.filter(isRankableIr).map(r=>r.ir);

    return [
      cat,
      median(ir3),
      pctPositive(ir3),
      median(w3.map(r=>r.alpha !== null ? +(r.alpha*100).toFixed(2) : null)),
      median(w3.map(r=>r.sharpe)),
      w3.length,
      median(ir1),
      pctPositive(ir1),
      median(ir5),
      pctPositive(ir5),
      median(irAll),
      pctPositive(irAll),
    ];
  });

  if (dataOut.length > 0) {
    sheet.getRange(3, 1, dataOut.length, HEADERS.length).setValues(dataOut);
  }

  const headerRange = sheet.getRange(2, 1, 1, HEADERS.length);
  headerRange.setFontWeight("bold").setBackground("#3F51B5").setFontColor("#FFFFFF");
  sheet.setFrozenRows(2);

  // Colour 3Y Median IR and % IR>0 columns
  if (dataOut.length > 0) {
    const medIrRange = sheet.getRange(3, 2, dataOut.length, 1);
    const pctIrRange = sheet.getRange(3, 3, dataOut.length, 1);
    applyIrFormatting(sheet, medIrRange);
    applyPctIrPositiveFormatting(sheet, pctIrRange);
  }

  sheet.autoResizeColumns(1, HEADERS.length);
  Logger.log("Category Summary: " + dataOut.length + " categories.");
}


// ---------------------------------------------------------------------------
// Tab 4: TFI Summary
// ---------------------------------------------------------------------------

function buildTfiSummarySheet(ss, records, runDate) {
  const sheet = getOrCreateSheet(ss, "TFI Summary");
  clearSheet(sheet);

  // 3Y window only for TFI comparison
  const r3y = records.filter(r => r.window === "3y" && r.fetch_ok && r.ir !== null && isRankableIr(r));

  const byTfi = {};
  r3y.forEach(r => {
    if (!byTfi[r.tfi_name]) byTfi[r.tfi_name] = [];
    byTfi[r.tfi_name].push(r);
  });

  function median(arr) {
    const sorted = arr.filter(v=>v!==null).sort((a,b)=>a-b);
    if (!sorted.length) return null;
    const mid = Math.floor(sorted.length/2);
    return sorted.length%2 ? sorted[mid] : (sorted[mid-1]+sorted[mid])/2;
  }

  const HEADERS = [
    "TFI", "Funds (3Y KID)", "Median IR (KID)", "% Funds IR>0 (KID)",
    "Median Alpha%", "Median Sharpe", "Median Fee%",
    "Categories covered"
  ];

  sheet.appendRow(["TFI Summary (3Y window)  |  Run: " + runDate]);
  sheet.appendRow(HEADERS);

  const tfis = Object.keys(byTfi).sort();
  const dataOut = tfis.map(tfi => {
    const funds = byTfi[tfi];
    const irs   = funds.map(r=>r.ir);
    const cats  = [...new Set(funds.map(r=>r.category))].sort().join(", ");
    return [
      tfi,
      funds.length,
      median(irs),
      +(funds.filter(r=>r.ir>0).length / funds.length).toFixed(3),
      median(funds.map(r=>r.alpha !== null ? +(r.alpha*100).toFixed(2) : null)),
      median(funds.map(r=>r.sharpe)),
      median(funds.map(r=>r.fee !== null ? +(r.fee*100).toFixed(2) : null)),
      cats,
    ];
  });

  if (dataOut.length > 0) {
    sheet.getRange(3, 1, dataOut.length, HEADERS.length).setValues(dataOut);
  }

  const headerRange = sheet.getRange(2, 1, 1, HEADERS.length);
  headerRange.setFontWeight("bold").setBackground("#3F51B5").setFontColor("#FFFFFF");
  sheet.setFrozenRows(2);

  if (dataOut.length > 0) {
    const medIrRange = sheet.getRange(3, 3, dataOut.length, 1);
    const pctIrRange = sheet.getRange(3, 4, dataOut.length, 1);
    applyIrFormatting(sheet, medIrRange);
    applyPctIrPositiveFormatting(sheet, pctIrRange);
  }

  sheet.autoResizeColumns(1, HEADERS.length);
  Logger.log("TFI Summary: " + dataOut.length + " TFIs.");
}


// ---------------------------------------------------------------------------
// Tab 5: Window Comparison — one row per fund, IR across all windows
// ---------------------------------------------------------------------------

function buildWindowComparisonSheet(ss, records, runDate) {
  const sheet = getOrCreateSheet(ss, "Window Comparison");
  clearSheet(sheet);

  // Pivot: one row per (stooq_id), columns for each window
  const byFund = {};
  records.forEach(r => {
    if (!r.fetch_ok) return;
    if (!byFund[r.stooq_id]) {
      byFund[r.stooq_id] = {
        stooq_id:  r.stooq_id,
        knf_name:  r.knf_name,
        tfi_name:  r.tfi_name,
        category:  r.category,
        fee:       r.fee,
        benchmark_quality: r.benchmark_quality,
        benchmark_source: r.benchmark_source,
        benchmark_coverage: r.benchmark_coverage,
      };
    }
    byFund[r.stooq_id][r.window] = r;
  });

  const HEADERS = [
    "Category", "Fund Name", "TFI", "Fee %",
    // For each window: IR, alpha%, sharpe, pct_ir_positive
    "1Y IR", "1Y Alpha%", "1Y Sharpe", "1Y % Win>0",
    "3Y IR", "3Y Alpha%", "3Y Sharpe", "3Y % Win>0",
    "5Y IR", "5Y Alpha%", "5Y Sharpe", "5Y % Win>0",
    "All IR", "All Alpha%", "All Sharpe", "All % Win>0",
    // Consistency: count of windows with IR > 0
    "# Windows IR>0 (KID)", "Benchmark Quality", "Benchmark Source", "Coverage",
  ];

  sheet.appendRow(["Window Comparison  |  Run: " + runDate]);
  sheet.appendRow(HEADERS);

  function w(fund, window, field) {
    const r = fund[window];
    if (!r) return null;
    if (field === "ir" && !isRankableIr(r)) return null;
    const v = r[field];
    if (v === null || v === undefined) return null;
    if (field === "alpha") return +(v * 100).toFixed(2);
    if (field === "pct_ir_pos") return v;
    return v;
  }

  const funds = Object.values(byFund)
    .sort((a,b) => {
      if (a.category < b.category) return -1;
      if (a.category > b.category) return 1;
      // Within category, sort by 3Y IR descending
      const ir3a = a["3y"] ? (a["3y"].ir || -9999) : -9999;
      const ir3b = b["3y"] ? (b["3y"].ir || -9999) : -9999;
      return ir3b - ir3a;
    });

  const dataOut = funds.map(fund => {
    const windows = ["1y", "3y", "5y", "all"];
    const nPositive = windows.filter(wn => {
      const r = fund[wn];
      return r && isRankableIr(r) && r.ir !== null && r.ir > 0;
    }).length;

    return [
      fund.category,
      fund.knf_name,
      fund.tfi_name,
      fund.fee !== null ? +(fund.fee * 100).toFixed(2) : null,
      w(fund,"1y","ir"),   w(fund,"1y","alpha"), w(fund,"1y","sharpe"), w(fund,"1y","pct_ir_pos"),
      w(fund,"3y","ir"),   w(fund,"3y","alpha"), w(fund,"3y","sharpe"), w(fund,"3y","pct_ir_pos"),
      w(fund,"5y","ir"),   w(fund,"5y","alpha"), w(fund,"5y","sharpe"), w(fund,"5y","pct_ir_pos"),
      w(fund,"all","ir"),  w(fund,"all","alpha"), w(fund,"all","sharpe"), w(fund,"all","pct_ir_pos"),
      nPositive,
      fund.benchmark_quality,
      fund.benchmark_source,
      fund.benchmark_coverage,
    ];
  });

  if (dataOut.length > 0) {
    sheet.getRange(3, 1, dataOut.length, HEADERS.length).setValues(dataOut);
  }

  const headerRange = sheet.getRange(2, 1, 1, HEADERS.length);
  headerRange.setFontWeight("bold").setBackground("#3F51B5").setFontColor("#FFFFFF");
  sheet.setFrozenRows(2);
  sheet.setFrozenColumns(2);

  // Apply IR formatting to each of the 4 IR columns: cols 5, 9, 13, 17
  if (dataOut.length > 0) {
    [5, 9, 13, 17].forEach(col => {
      const range = sheet.getRange(3, col, dataOut.length, 1);
      applyIrFormatting(sheet, range);
    });
    // # Windows IR>0 column (last): colour 4 = all green, 0 = red
    const consistencyRange = sheet.getRange(3, 21, dataOut.length, 1);
    const rules = sheet.getConditionalFormatRules();
    rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenNumberGreaterThanOrEqualTo(3)
      .setBackground("#B7E1CD")
      .setRanges([consistencyRange])
      .build());
    rules.push(SpreadsheetApp.newConditionalFormatRule()
      .whenNumberEqualTo(0)
      .setBackground("#FFDEDE")
      .setRanges([consistencyRange])
      .build());
    sheet.setConditionalFormatRules(rules);
  }

  sheet.autoResizeColumns(1, HEADERS.length);
  Logger.log("Window Comparison: " + dataOut.length + " funds.");
}


// ---------------------------------------------------------------------------
// Trigger setup — run once manually
// ---------------------------------------------------------------------------

function setupTrigger() {
  // Remove any existing triggers for this function to avoid duplicates
  ScriptApp.getProjectTriggers()
    .filter(t => t.getHandlerFunction() === "importFundPerformance")
    .forEach(t => ScriptApp.deleteTrigger(t));

  // Daily at 21:00 Warsaw — the script picks up the latest fund_ranking_YYYYMMDD.csv
  // by filename, so on days when no new file was uploaded it simply re-imports
  // the existing one. Harmless, and ensures the sheet is always current.
  ScriptApp.newTrigger("importFundPerformance")
    .timeBased()
    .atHour(21)
    .everyDays(1)
    .inTimezone("Europe/Warsaw")
    .create();

  Logger.log("Trigger registered: importFundPerformance daily at 21:00 Warsaw.");
}