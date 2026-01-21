from __future__ import annotations

import calendar
import hashlib
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import polars as pl
except Exception:
    pl = None


APP_TITLE = "Rekon Otomatis (Payment vs Settlement vs Rekening Koran)"
CACHE_DIR = Path(".cache_uploads")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities: normalization
# -----------------------------
def _norm_col(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm_to_real = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_to_real:
            return norm_to_real[key]
    return None


def _clean_amount_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    out = s.astype(str)
    out = out.str.replace(r"[\s]", "", regex=True)
    out = out.str.replace(".", "", regex=False)
    out = out.str.replace(",", ".", regex=False)
    out = out.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(out, errors="coerce").fillna(0.0)


def _to_datetime_series(s: pd.Series, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, infer_datetime_format=True)


def _month_range(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=month + 1, day=1)
    return start, end


def _contains_ci(series: pd.Series, needle: str) -> pd.Series:
    return series.astype(str).str.contains(needle, case=False, na=False)


def _contains_any_ci(series: pd.Series, needles: Iterable[str]) -> pd.Series:
    s = series.astype(str)
    mask = pd.Series(False, index=series.index)
    for n in needles:
        mask |= s.str.contains(n, case=False, na=False)
    return mask


def _safe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -----------------------------
# Upload persistence (stream to disk + hash)
# -----------------------------
@dataclass(frozen=True)
class PersistedFile:
    sha256: str
    path: Path
    orig_name: str


def persist_upload(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> PersistedFile:
    hasher = hashlib.sha256()
    orig = Path(uploaded.name).name
    suffix = "".join(Path(orig).suffixes) or ""
    tmp_path = CACHE_DIR / f"tmp_{int(time.time() * 1e6)}{suffix}"

    with tmp_path.open("wb") as f:
        while True:
            chunk = uploaded.read(8 * 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            f.write(chunk)

    sha = hasher.hexdigest()
    final_path = CACHE_DIR / f"{sha}_{orig}"
    if final_path.exists():
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.replace(final_path)

    uploaded.seek(0)
    return PersistedFile(sha256=sha, path=final_path, orig_name=orig)


def persist_uploads(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[PersistedFile]:
    return [persist_upload(f) for f in files]


def extract_all_from_zip(zip_path: Path, wanted_exts: Tuple[str, ...]) -> List[Path]:
    out_paths: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        candidates = [n for n in names if n.lower().endswith(tuple(e.lower() for e in wanted_exts))]
        if not candidates:
            raise ValueError(f"Zip tidak berisi file {wanted_exts}. Isi: {names[:40]}")

        for chosen in sorted(candidates, key=lambda x: (len(x), x)):
            raw = zf.read(chosen)
            sha = hashlib.sha256(raw).hexdigest()
            out_name = Path(chosen).name
            out_path = CACHE_DIR / f"{sha}_{out_name}"
            if not out_path.exists():
                out_path.write_bytes(raw)
            out_paths.append(out_path)

    return out_paths


# -----------------------------
# Readers (optimized columns)
# -----------------------------
def read_excel_any(path: Path, sheet_name: Optional[str], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".xlsb":
        return pd.read_excel(path, sheet_name=sheet_name or 0, engine="pyxlsb", usecols=usecols)
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name=sheet_name or 0, engine="openpyxl", usecols=usecols)
    if ext == ".xls":
        return pd.read_excel(path, sheet_name=sheet_name or 0, engine="xlrd", usecols=usecols)
    raise ValueError(f"Unsupported excel extension: {ext}")


def read_csv_fast(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if pl is not None:
        try:
            lf = pl.scan_csv(str(path), ignore_errors=True, infer_schema_length=2000)
            if columns:
                lf = lf.select([c for c in columns if c in lf.columns])
            return lf.collect(streaming=True).to_pandas()
        except Exception:
            pass
    return pd.read_csv(path, usecols=columns, low_memory=False, encoding_errors="ignore")


# -----------------------------
# Payment Report parsing + Table 1
# -----------------------------
PAYMENT_EXPECTED = {
    "tanggal": ["Tanggal Pembayaran", "TANGGAL PEMBAYARAN", "TANGGAL"],
    "asal": ["ASAL", "PELAbuhan", "PELABUHAN", "CABANG"],
    "tipe": ["TIPE PEMBAYARAN", "TIPE_PEMBAYARAN", "PAYMENT TYPE"],
    "sof": ["SOF ID", "SOF_ID", "SOFID"],
    "amount": ["TOTAL TARIF (Rp.)", "TOTAL TARIF (RP.)", "TOTALTARIFRP", "TOTAL TARIF"],
    "invoice": ["Nomor Invoice", "NO INVOICE", "INVOICE", "NOMORINVOICE"],
}


def parse_payment_report(
    path: Path,
    sheet_name: Optional[str],
    year: int,
    month: int,
    dayfirst: bool,
    asal_split_mode: str,
) -> pd.DataFrame:
    df = read_excel_any(path, sheet_name=sheet_name, usecols=None)
    cols = {k: _find_col(df, v) for k, v in PAYMENT_EXPECTED.items()}
    missing = [k for k, c in cols.items() if c is None and k in ("tanggal", "asal", "tipe", "sof", "amount")]
    if missing:
        raise ValueError(f"Kolom wajib Payment Report tidak ketemu: {missing}. Kolom ada: {list(df.columns)[:50]}")

    out = pd.DataFrame()
    out["tanggal"] = _to_datetime_series(df[cols["tanggal"]], dayfirst=dayfirst)
    out["asal_raw"] = df[cols["asal"]].astype(str)
    out["tipe_pembayaran"] = df[cols["tipe"]].astype(str)
    out["sof_id"] = df[cols["sof"]].astype(str)
    out["amount"] = _clean_amount_series(df[cols["amount"]])
    out["nomor_invoice"] = df[cols["invoice"]].astype(str) if cols["invoice"] is not None else ""

    if asal_split_mode == "left":
        out["pelabuhan"] = out["asal_raw"].str.split("-", n=1, expand=True)[0].str.strip()
    elif asal_split_mode == "right":
        out["pelabuhan"] = out["asal_raw"].str.split("-", n=1, expand=True).iloc[:, -1].str.strip()
    else:
        out["pelabuhan"] = out["asal_raw"].str.strip()

    start, end = _month_range(year, month)
    out = out[(out["tanggal"] >= start) & (out["tanggal"] < end)].copy()
    out["pelabuhan"] = out["pelabuhan"].replace("", "UNKNOWN").fillna("UNKNOWN")
    return out


def build_table_1_payment_detail(payment: pd.DataFrame) -> pd.DataFrame:
    tipe = payment["tipe_pembayaran"].astype(str)
    sof = payment["sof_id"].astype(str)

    is_cash = _contains_ci(tipe, "cash")
    is_pre_bri = _contains_ci(tipe, "prepaid-bri")
    is_pre_bni = _contains_ci(tipe, "prepaid-bni")
    is_pre_mandiri = _contains_ci(tipe, "prepaid-mandiri")
    is_pre_bca = _contains_ci(tipe, "prepaid-bca")
    is_skpt = _contains_ci(tipe, "skpt")
    is_ifcs = _contains_ci(tipe, "ifcs")
    is_reedem = _contains_ci(tipe, "reedem")

    is_finpay = _contains_ci(tipe, "finpay")
    is_espay = is_finpay & _contains_ci(sof, "spay")
    is_finnet = is_finpay & _contains_ci(sof, "finpay021")

    is_bca = _contains_any_ci(sof, ["bca", "blu"])
    is_non_bca = ~is_bca

    is_non = is_cash | is_pre_bri | is_pre_bni | is_pre_mandiri | is_pre_bca | is_reedem | is_skpt | is_ifcs

    def _sum(mask: pd.Series) -> pd.Series:
        return payment["amount"].where(mask, 0.0)

    tmp = payment[["tanggal", "pelabuhan"]].copy()
    tmp["Cash"] = _sum(is_cash)
    tmp["Prepaid BRI"] = _sum(is_pre_bri)
    tmp["Prepaid BNI"] = _sum(is_pre_bni)
    tmp["Prepaid Mandiri"] = _sum(is_pre_mandiri)
    tmp["Prepaid BCA"] = _sum(is_pre_bca)
    tmp["SKPT"] = _sum(is_skpt)
    tmp["IFCS"] = _sum(is_ifcs)
    tmp["Reedem"] = _sum(is_reedem)
    tmp["ESPAY"] = _sum(is_espay)
    tmp["FINNET"] = _sum(is_finnet)

    cols_a_l = [
        "Cash",
        "Prepaid BRI",
        "Prepaid BNI",
        "Prepaid Mandiri",
        "Prepaid BCA",
        "SKPT",
        "IFCS",
        "Reedem",
        "ESPAY",
        "FINNET",
    ]
    tmp["Total (a-l)"] = tmp[cols_a_l].sum(axis=1)

    tmp["BCA"] = _sum(is_bca)
    tmp["NON BCA"] = _sum(is_non_bca)
    tmp["NON"] = _sum(is_non)
    tmp["Total (n-p)"] = tmp[["BCA", "NON BCA", "NON"]].sum(axis=1)

    out = (
        tmp.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .sum(numeric_only=True)
        .sort_values(["tanggal", "pelabuhan"])
    )
    return out


# -----------------------------
# Settlement ESPAY parsing + Table 2
# -----------------------------
ESPAY_EXPECTED = {
    "tanggal": ["Date", "DATE", "Tanggal", "Transaction Date"],
    "merchant": ["Merchant_Name", "MERCHANT_NAME", "Merchant Name", "VA Name", "VA_NAME"],
    "va_name": ["VA Name", "VA_NAME"],
    "channel": ["Channel", "CHANNEL"],
    "amount": ["Amount", "AMOUNT"],
    "fee": ["Tx Fee", "TX FEE", "Fee", "TXFEE"],
    "settlement_amount": ["Settlement Amount", "SETTLEMENT AMOUNT"],
    "bank": ["Bank", "BANK"],
    "product": ["Product Name", "PRODUCT NAME", "PRODUCT"],
}


def parse_settlement_espay(path: Path, sheet_name: Optional[str], year: int, month: int, dayfirst: bool) -> pd.DataFrame:
    ext = path.suffix.lower()
    df = read_csv_fast(path, columns=None) if ext == ".csv" else read_excel_any(path, sheet_name=sheet_name, usecols=None)

    cols = {k: _find_col(df, v) for k, v in ESPAY_EXPECTED.items()}
    if cols["tanggal"] is None:
        raise ValueError("Kolom tanggal Settlement ESPAY tidak ditemukan (Date).")

    tanggal = _to_datetime_series(df[cols["tanggal"]], dayfirst=dayfirst)

    merchant_col = cols["merchant"] or cols["va_name"]
    if merchant_col is None:
        raise ValueError("Kolom Pelabuhan Settlement ESPAY tidak ditemukan (Merchant_Name/VA Name).")

    channel_col = cols["channel"] or ""
    bank_col = cols["bank"] or ""
    product_col = cols["product"] or ""

    amount_col = cols["settlement_amount"] or cols["amount"]
    if amount_col is None:
        raise ValueError("Kolom amount Settlement ESPAY tidak ditemukan (Settlement Amount / Amount).")

    if cols["settlement_amount"] is not None:
        net_amount = _clean_amount_series(df[cols["settlement_amount"]])
    else:
        amt = _clean_amount_series(df[cols["amount"]]) if cols["amount"] is not None else 0.0
        fee = _clean_amount_series(df[cols["fee"]]) if cols["fee"] is not None else 0.0
        net_amount = amt - fee

    out = pd.DataFrame(
        {
            "tanggal": tanggal,
            "pelabuhan": df[merchant_col].astype(str).str.strip().replace("", "UNKNOWN"),
            "channel": df[channel_col].astype(str) if channel_col else "",
            "bank": df[bank_col].astype(str) if bank_col else "",
            "product": df[product_col].astype(str) if product_col else "",
            "net_amount": net_amount.fillna(0.0),
        }
    )

    start, end = _month_range(year, month)
    out = out[(out["tanggal"] >= start) & (out["tanggal"] < end)].copy()
    out["pelabuhan"] = out["pelabuhan"].fillna("UNKNOWN")
    return out


def build_table_2_espay_detail(espay: pd.DataFrame) -> pd.DataFrame:
    ch = espay["channel"].astype(str)
    is_va = _contains_ci(ch, "va")
    is_emoney = ~is_va

    is_bca = (
        _contains_ci(espay["bank"].astype(str), "bca")
        | _contains_ci(espay["product"].astype(str), "bca")
        | _contains_ci(espay["channel"].astype(str), "bca")
    )
    is_non_bca = ~is_bca

    tmp = espay[["tanggal", "pelabuhan"]].copy()
    tmp["Virtual Account"] = espay["net_amount"].where(is_va, 0.0)
    tmp["E-Money"] = espay["net_amount"].where(is_emoney, 0.0)
    tmp["Total VA + E-Money"] = tmp[["Virtual Account", "E-Money"]].sum(axis=1)

    tmp["BCA"] = espay["net_amount"].where(is_bca, 0.0)
    tmp["NON BCA"] = espay["net_amount"].where(is_non_bca, 0.0)
    tmp["Total BCA + NON BCA"] = tmp[["BCA", "NON BCA"]].sum(axis=1)

    tmp["n_tx"] = 1

    out = (
        tmp.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .sum(numeric_only=True)
        .sort_values(["tanggal", "pelabuhan"])
    )
    return out


# -----------------------------
# Settlement FINNET parsing + Table 3
# -----------------------------
FINNET_EXPECTED = {
    "tanggal": ["Payment Date Time", "PAYMENT DATE TIME", "Payment Date", "DATE"],
    "merchant": ["Merchant Name", "MERCHANT NAME", "Merchant_Name", "MERCHANT_NAME"],
    "method": ["Payment Method", "PAYMENT METHOD", "Method"],
    "amount": ["Merchant Amount", "MERCHANT AMOUNT", "Amount"],
}


def parse_settlement_finnet(path: Path, sheet_name: Optional[str], year: int, month: int, dayfirst: bool) -> pd.DataFrame:
    ext = path.suffix.lower()
    df = read_csv_fast(path, columns=None) if ext == ".csv" else read_excel_any(path, sheet_name=sheet_name, usecols=None)

    cols = {k: _find_col(df, v) for k, v in FINNET_EXPECTED.items()}
    for k in ("tanggal", "merchant", "method", "amount"):
        if cols[k] is None:
            raise ValueError(f"Kolom wajib Settlement FINNET tidak ditemukan: {k}")

    out = pd.DataFrame()
    out["tanggal"] = _to_datetime_series(df[cols["tanggal"]], dayfirst=dayfirst)
    out["pelabuhan"] = df[cols["merchant"]].astype(str).str.strip().replace("", "UNKNOWN")
    out["payment_method"] = df[cols["method"]].astype(str)
    out["amount"] = _clean_amount_series(df[cols["amount"]]).fillna(0.0)

    start, end = _month_range(year, month)
    out = out[(out["tanggal"] >= start) & (out["tanggal"] < end)].copy()
    out["pelabuhan"] = out["pelabuhan"].fillna("UNKNOWN")
    return out


def build_table_3_finnet_detail(finnet: pd.DataFrame) -> pd.DataFrame:
    pm = finnet["payment_method"].astype(str)
    is_va = _contains_ci(pm, "va")
    is_emoney = ~is_va

    is_bca = _contains_any_ci(pm, ["bca", "blu"])
    is_non_bca = ~is_bca

    tmp = finnet[["tanggal", "pelabuhan"]].copy()
    tmp["Virtual Account"] = finnet["amount"].where(is_va, 0.0)
    tmp["E-Money"] = finnet["amount"].where(is_emoney, 0.0)
    tmp["Total VA + E-Money"] = tmp[["Virtual Account", "E-Money"]].sum(axis=1)

    tmp["BCA"] = finnet["amount"].where(is_bca, 0.0)
    tmp["NON BCA"] = finnet["amount"].where(is_non_bca, 0.0)
    tmp["Total BCA + NON BCA"] = tmp[["BCA", "NON BCA"]].sum(axis=1)

    tmp["n_tx"] = 1

    out = (
        tmp.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .sum(numeric_only=True)
        .sort_values(["tanggal", "pelabuhan"])
    )
    return out


# -----------------------------
# Bank statement parsing
# -----------------------------
def _detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["Tanggal", "TANGGAL", "Date", "DATE", "Txn Date", "Transaction Date"]:
        c = _find_col(df, [cand])
        if c is not None:
            return c
    return df.columns[0] if len(df.columns) else None


def _detect_text_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    c = _find_col(df, preferred)
    if c is not None:
        return c
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    return df.columns[0] if len(df.columns) else None


def _detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "Kredit",
        "KREDIT",
        "Credit",
        "CREDIT",
        "Mutasi Kredit",
        "MUTASI KREDIT",
        "CR",
        "Amount",
        "AMOUNT",
        "Nominal",
        "NOMINAL",
        "Jumlah",
        "JUMLAH",
    ]
    c = _find_col(df, preferred)
    if c is not None:
        return c
    best = None
    best_score = -1.0
    for col in df.columns:
        s = _clean_amount_series(df[col])
        score = (s.notna() & (s != 0)).mean()
        if score > best_score:
            best_score = score
            best = col
    return best


def parse_bank_statement(
    path: Path,
    sheet_name: Optional[str],
    year: int,
    month: int,
    dayfirst: bool,
    text_candidates: List[str],
    keyword_filters: List[str],
    date_plus_days: int = 1,
) -> pd.DataFrame:
    df = read_excel_any(path, sheet_name=sheet_name, usecols=None)
    date_col = _detect_date_col(df)
    text_col = _detect_text_col(df, text_candidates)
    amt_col = _detect_amount_col(df)

    if date_col is None or text_col is None or amt_col is None:
        raise ValueError("Gagal auto-detect kolom rekening koran (tanggal/text/amount).")

    out = pd.DataFrame()
    out["tanggal_raw"] = _to_datetime_series(df[date_col], dayfirst=dayfirst)
    out["text"] = df[text_col].astype(str)
    out["amount"] = _clean_amount_series(df[amt_col]).fillna(0.0)

    mask = _contains_any_ci(out["text"], keyword_filters)
    out = out[mask].copy()

    out["tanggal"] = out["tanggal_raw"] + pd.to_timedelta(date_plus_days, unit="D")
    start, end = _month_range(year, month)
    out = out[(out["tanggal"] >= start) & (out["tanggal"] < end)].copy()

    out["n_tx"] = 1
    return out[["tanggal", "text", "amount", "n_tx"]]


def assign_pelabuhan_from_text(bank_df: pd.DataFrame, pelabuhan_list: List[str]) -> pd.DataFrame:
    if bank_df.empty:
        bank_df = bank_df.copy()
        bank_df["pelabuhan"] = []
        return bank_df

    pelabuhan_list = [p for p in pelabuhan_list if isinstance(p, str) and p.strip()]
    if not pelabuhan_list:
        out = bank_df.copy()
        out["pelabuhan"] = "ALL"
        return out

    patterns = [(p, re.compile(re.escape(p), re.IGNORECASE)) for p in pelabuhan_list]

    def _match_one(txt: str) -> str:
        for p, pat in patterns:
            if pat.search(txt or ""):
                return p
        return "ALL"

    out = bank_df.copy()
    out["pelabuhan"] = out["text"].map(_match_one)
    return out


# -----------------------------
# Rekonsiliasi builders
# -----------------------------
def _group_amount_count(df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["tanggal", "pelabuhan", "amount", "n_tx"])
    return (
        df.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .agg(amount=(amount_col, "sum"), n_tx=("n_tx", "sum"))
    )


def build_recon_espay(payment: pd.DataFrame, espay_raw: pd.DataFrame, bank_bca: pd.DataFrame, bank_non_bca: pd.DataFrame) -> pd.DataFrame:
    sof = payment["sof_id"].astype(str)
    is_spay = _contains_ci(sof, "spay")
    is_bca = _contains_any_ci(sof, ["bca", "blu"])

    pay = payment.copy()
    pay["n_tx"] = (payment["nomor_invoice"].astype(str).str.len() > 0).astype(int)
    pay_spay = pay[is_spay].copy()

    pay_spay["tiket_bca"] = pay_spay["amount"].where(is_bca[is_spay], 0.0)
    pay_spay["tiket_non_bca"] = pay_spay["amount"].where(~is_bca[is_spay], 0.0)
    pay_spay["tiket_total"] = pay_spay["tiket_bca"] + pay_spay["tiket_non_bca"]

    pay_g = (
        pay_spay.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .agg(
            tiket_detail_bca=("tiket_bca", "sum"),
            tiket_detail_non_bca=("tiket_non_bca", "sum"),
            total_tiket_detail=("tiket_total", "sum"),
            jml_tiket_detail=("n_tx", "sum"),
        )
    )

    ch = espay_raw["channel"].astype(str)
    is_bca_s = (
        _contains_ci(espay_raw["bank"].astype(str), "bca")
        | _contains_ci(espay_raw["product"].astype(str), "bca")
        | _contains_ci(ch, "bca")
    )
    setl = espay_raw.copy()
    setl["settle_bca"] = setl["net_amount"].where(is_bca_s, 0.0)
    setl["settle_non_bca"] = setl["net_amount"].where(~is_bca_s, 0.0)
    setl["settle_total"] = setl["net_amount"]
    setl["n_tx"] = 1

    setl_g = (
        setl.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .agg(
            settlement_bca=("settle_bca", "sum"),
            settlement_non_bca=("settle_non_bca", "sum"),
            total_settlement_report=("settle_total", "sum"),
            jml_settlement_report=("n_tx", "sum"),
        )
    )

    bank_bca_g = _group_amount_count(bank_bca, "amount").rename(columns={"amount": "dana_masuk_bca", "n_tx": "jml_dana_masuk_bca"})
    bank_non_bca_g = _group_amount_count(bank_non_bca, "amount").rename(
        columns={"amount": "dana_masuk_non_bca", "n_tx": "jml_dana_masuk_non_bca"}
    )

    out = pay_g.merge(setl_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.merge(bank_bca_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.merge(bank_non_bca_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.fillna(0.0)

    out["total_dana_masuk"] = out["dana_masuk_bca"] + out["dana_masuk_non_bca"]
    out["jml_dana_masuk"] = out["jml_dana_masuk_bca"] + out["jml_dana_masuk_non_bca"]

    out["selisih_tiket_vs_settlement"] = out["total_tiket_detail"] - out["total_settlement_report"]
    out["selisih_dana_vs_settlement"] = out["total_dana_masuk"] - out["total_settlement_report"]

    return out.sort_values(["tanggal", "pelabuhan"])


def build_recon_finnet(payment: pd.DataFrame, finnet_raw: pd.DataFrame, bank_bca: pd.DataFrame, bank_non_bca: pd.DataFrame) -> pd.DataFrame:
    sof = payment["sof_id"].astype(str)
    is_spay = _contains_ci(sof, "spay")
    is_bca = _contains_any_ci(sof, ["bca", "blu"])

    pay = payment.copy()
    pay["n_tx"] = (payment["nomor_invoice"].astype(str).str.len() > 0).astype(int)
    pay_fn = pay[~is_spay].copy()

    pay_fn["tiket_bca"] = pay_fn["amount"].where(is_bca[~is_spay], 0.0)
    pay_fn["tiket_non_bca"] = pay_fn["amount"].where(~is_bca[~is_spay], 0.0)
    pay_fn["tiket_total"] = pay_fn["tiket_bca"] + pay_fn["tiket_non_bca"]

    pay_g = (
        pay_fn.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .agg(
            tiket_detail_bca=("tiket_bca", "sum"),
            tiket_detail_non_bca=("tiket_non_bca", "sum"),
            total_tiket_detail=("tiket_total", "sum"),
            jml_tiket_detail=("n_tx", "sum"),
        )
    )

    pm = finnet_raw["payment_method"].astype(str)
    is_bca_s = _contains_any_ci(pm, ["bca", "blu"])

    setl = finnet_raw.copy()
    setl["settle_bca"] = setl["amount"].where(is_bca_s, 0.0)
    setl["settle_non_bca"] = setl["amount"].where(~is_bca_s, 0.0)
    setl["settle_total"] = setl["amount"]
    setl["n_tx"] = 1

    setl_g = (
        setl.groupby(["tanggal", "pelabuhan"], as_index=False, sort=False)
        .agg(
            settlement_bca=("settle_bca", "sum"),
            settlement_non_bca=("settle_non_bca", "sum"),
            total_settlement_report=("settle_total", "sum"),
            jml_settlement_report=("n_tx", "sum"),
        )
    )

    bank_bca_g = _group_amount_count(bank_bca, "amount").rename(columns={"amount": "dana_masuk_bca", "n_tx": "jml_dana_masuk_bca"})
    bank_non_bca_g = _group_amount_count(bank_non_bca, "amount").rename(
        columns={"amount": "dana_masuk_non_bca", "n_tx": "jml_dana_masuk_non_bca"}
    )

    out = pay_g.merge(setl_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.merge(bank_bca_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.merge(bank_non_bca_g, on=["tanggal", "pelabuhan"], how="outer")
    out = out.fillna(0.0)

    out["total_dana_masuk"] = out["dana_masuk_bca"] + out["dana_masuk_non_bca"]
    out["jml_dana_masuk"] = out["jml_dana_masuk_bca"] + out["jml_dana_masuk_non_bca"]

    out["selisih_tiket_vs_settlement"] = out["total_tiket_detail"] - out["total_settlement_report"]
    out["selisih_dana_vs_settlement"] = out["total_dana_masuk"] - out["total_settlement_report"]

    return out.sort_values(["tanggal", "pelabuhan"])


# -----------------------------
# Summary tables + export
# -----------------------------
def build_summary(recon: pd.DataFrame, periode: str) -> pd.DataFrame:
    if recon.empty:
        return pd.DataFrame()

    g = (
        recon.groupby(["pelabuhan"], as_index=False)
        .agg(
            menu_payment_jml_tx=("jml_tiket_detail", "sum"),
            menu_payment_nominal=("total_tiket_detail", "sum"),
            settlement_jml_tx=("jml_settlement_report", "sum"),
            settlement_nominal=("total_settlement_report", "sum"),
            rekening_koran=("total_dana_masuk", "sum"),
        )
    )
    g.insert(0, "periode", periode)
    g = g.rename(columns={"pelabuhan": "cabang"})

    g["selisih_menu_vs_settlement_jml_tx"] = g["menu_payment_jml_tx"] - g["settlement_jml_tx"]
    g["selisih_menu_vs_settlement_nominal"] = g["menu_payment_nominal"] - g["settlement_nominal"]
    g["selisih_dana_masuk_vs_settlement"] = g["rekening_koran"] - g["settlement_nominal"]

    return g.sort_values(["cabang"])


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    return bio.getvalue()


# -----------------------------
# Streamlit App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Parameter Periode")
        now = datetime.now()
        month = st.selectbox("Month", list(range(1, 13)), index=now.month - 1, format_func=lambda m: calendar.month_name[m])
        year = st.selectbox("Year", list(range(now.year - 5, now.year + 1)), index=5)

        st.header("Opsi Parsing")
        dayfirst = st.toggle("dayfirst (DD/MM/YYYY)", value=True)
        asal_split_mode = st.selectbox("ASAL split", ["left", "right", "full"], index=0)

        st.header("Upload Files (Multiple)")
        up_payment = st.file_uploader(
            "Payment Report (xlsb / zip(xlsb))",
            type=["xlsb", "zip"],
            accept_multiple_files=True,
        )
        up_espay = st.file_uploader(
            "Settlement ESPAY (csv/xlsx/xls)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        up_finnet = st.file_uploader(
            "Settlement FINNET (zip(csv)/csv/xlsx/xls)",
            type=["zip", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        up_bca = st.file_uploader(
            "Rekening Koran BCA (xls/xlsx)",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
        )
        up_non_bca = st.file_uploader(
            "Rekening Koran Non-BCA (xls/xlsx)",
            type=["xls", "xlsx"],
            accept_multiple_files=True,
        )

        process = st.button("Process", type="primary", use_container_width=True)

    if not process:
        st.info("Upload semua file (boleh banyak) lalu klik **Process**.")
        return

    if not all([up_payment, up_espay, up_finnet, up_bca, up_non_bca]):
        st.error("Semua kategori file wajib di-upload (boleh multiple).")
        return

    t0 = time.time()
    try:
        p_payments = persist_uploads(up_payment)
        p_espays = persist_uploads(up_espay)
        p_finnets = persist_uploads(up_finnet)
        p_bcas = persist_uploads(up_bca)
        p_non_bcas = persist_uploads(up_non_bca)

        # ---- Payment: parse many (zip may contain many xlsb)
        payment_parts: List[pd.DataFrame] = []
        with st.spinner("Parsing Payment Report (multiple)..."):
            for pf in p_payments:
                if pf.path.suffix.lower() == ".zip":
                    for xlsb_path in extract_all_from_zip(pf.path, wanted_exts=(".xlsb",)):
                        payment_parts.append(parse_payment_report(xlsb_path, None, year, month, dayfirst, asal_split_mode))
                else:
                    payment_parts.append(parse_payment_report(pf.path, None, year, month, dayfirst, asal_split_mode))
            payment_df = _safe_concat(payment_parts)
            if payment_df.empty:
                raise ValueError("Payment Report hasil parsing kosong setelah filter periode.")
            table1 = build_table_1_payment_detail(payment_df)

        # ---- ESPAY: parse many
        espay_parts: List[pd.DataFrame] = []
        with st.spinner("Parsing Settlement ESPAY (multiple)..."):
            for pf in p_espays:
                espay_parts.append(parse_settlement_espay(pf.path, None, year, month, dayfirst))
            espay_df = _safe_concat(espay_parts)
            table2 = build_table_2_espay_detail(espay_df) if not espay_df.empty else pd.DataFrame()

        # ---- FINNET: parse many (zip may contain many csv)
        finnet_parts: List[pd.DataFrame] = []
        with st.spinner("Parsing Settlement FINNET (multiple)..."):
            for pf in p_finnets:
                if pf.path.suffix.lower() == ".zip":
                    for csv_path in extract_all_from_zip(pf.path, wanted_exts=(".csv",)):
                        finnet_parts.append(parse_settlement_finnet(csv_path, None, year, month, dayfirst))
                else:
                    finnet_parts.append(parse_settlement_finnet(pf.path, None, year, month, dayfirst))
            finnet_df = _safe_concat(finnet_parts)
            table3 = build_table_3_finnet_detail(finnet_df) if not finnet_df.empty else pd.DataFrame()

        pelabuhan_list = sorted(set(payment_df["pelabuhan"].dropna().astype(str).tolist()))

        # ---- Bank statements: parse many then concat
        rk_bca_parts: List[pd.DataFrame] = []
        with st.spinner("Parsing Rekening Koran BCA (multiple)..."):
            for pf in p_bcas:
                rk_bca_parts.append(
                    parse_bank_statement(
                        pf.path,
                        None,
                        year,
                        month,
                        dayfirst,
                        text_candidates=["Keterangan", "KETERANGAN", "Uraian", "Description"],
                        keyword_filters=["SGW", "FINIF", "FINON"],
                        date_plus_days=1,
                    )
                )
            rk_bca = assign_pelabuhan_from_text(_safe_concat(rk_bca_parts), pelabuhan_list)

        rk_non_parts: List[pd.DataFrame] = []
        with st.spinner("Parsing Rekening Koran Non-BCA (multiple)..."):
            for pf in p_non_bcas:
                rk_non_parts.append(
                    parse_bank_statement(
                        pf.path,
                        None,
                        year,
                        month,
                        dayfirst,
                        text_candidates=["Remark", "REMARK", "Keterangan", "KETERANGAN", "Uraian", "Description"],
                        keyword_filters=["SGW", "FINIF", "FINON"],
                        date_plus_days=1,
                    )
                )
            rk_non = assign_pelabuhan_from_text(_safe_concat(rk_non_parts), pelabuhan_list)

        # Split RK keywords per recon type
        rk_bca_sgw = rk_bca[_contains_ci(rk_bca["text"], "sgw")].copy()
        rk_non_sgw = rk_non[_contains_ci(rk_non["text"], "sgw")].copy()

        rk_bca_fin = rk_bca[_contains_any_ci(rk_bca["text"], ["finif", "finon"])].copy()
        rk_non_fin = rk_non[_contains_any_ci(rk_non["text"], ["finif", "finon"])].copy()

        with st.spinner("Building Rekonsiliasi ESPAY..."):
            recon_espay = build_recon_espay(payment_df, espay_df, rk_bca_sgw, rk_non_sgw) if not espay_df.empty else pd.DataFrame()

        with st.spinner("Building Rekonsiliasi FINNET..."):
            recon_finnet = build_recon_finnet(payment_df, finnet_df, rk_bca_fin, rk_non_fin) if not finnet_df.empty else pd.DataFrame()

        periode = f"{calendar.month_name[month]} {year}"
        summary_espay = build_summary(recon_espay, periode=periode) if not recon_espay.empty else pd.DataFrame()
        summary_finnet = build_summary(recon_finnet, periode=periode) if not recon_finnet.empty else pd.DataFrame()

        elapsed = time.time() - t0
        st.success(f"Selesai diproses. Waktu: {elapsed:.1f} detik")

        tabs = st.tabs(
            [
                "1) Detail Payment",
                "2) Detail Settlement ESPAY",
                "3) Detail Settlement FINNET",
                "4) Rekonsiliasi ESPAY",
                "5) Rekonsiliasi FINNET",
                "6) Summary ESPAY",
                "7) Summary FINNET",
            ]
        )

        with tabs[0]:
            st.subheader("Table 1 - Detail Payment Report")
            st.dataframe(table1, use_container_width=True, height=520)

        with tabs[1]:
            st.subheader("Table 2 - Detail Settlement ESPAY")
            st.dataframe(table2, use_container_width=True, height=520)

        with tabs[2]:
            st.subheader("Table 3 - Detail Settlement FINNET")
            st.dataframe(table3, use_container_width=True, height=520)

        with tabs[3]:
            st.subheader("Table 4 - Rekonsiliasi ESPAY")
            st.dataframe(recon_espay, use_container_width=True, height=520)

        with tabs[4]:
            st.subheader("Table 5 - Rekonsiliasi FINNET")
            st.dataframe(recon_finnet, use_container_width=True, height=520)

        with tabs[5]:
            st.subheader("Table 6 - Summary ESPAY")
            st.dataframe(summary_espay, use_container_width=True, height=520)

        with tabs[6]:
            st.subheader("Table 7 - Summary FINNET")
            st.dataframe(summary_finnet, use_container_width=True, height=520)

        sheets = {
            "1_Detail_Payment": table1,
            "2_Detail_Settlement_ESPAY": table2,
            "3_Detail_Settlement_FINNET": table3,
            "4_Rekonsiliasi_ESPAY": recon_espay,
            "5_Rekonsiliasi_FINNET": recon_finnet,
            "6_Summary_ESPAY": summary_espay,
            "7_Summary_FINNET": summary_finnet,
        }
        xlsx_bytes = to_excel_bytes(sheets)
        st.download_button(
            "Download Excel (All Tables)",
            data=xlsx_bytes,
            file_name=f"rekon_{year}-{month:02d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    except Exception as e:
        st.exception(e)


if __name__ == "__main__":
    main()
