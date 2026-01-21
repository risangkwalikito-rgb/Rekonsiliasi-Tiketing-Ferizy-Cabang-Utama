# app.py
"""
Rekonsiliasi Payment Report (XLSB) - Table only

Install:
  pip install streamlit pandas pyxlsb

Required columns (case-insensitive, tolerant spaces/punctuation):
- TOTAL TARIF (Rp.)
- Tanggal Pembayaran
- ASAL
- TIPE PEMBAYARAN
- SOF ID

Business rule (exclusive buckets to avoid double-counting):
- NON     := TIPE PEMBAYARAN in {cash, prepaid-*, reedem, skpt, ifcs}
- BCA     := NOT NON AND SOF ID contains {bca, blu}
- NON BCA := NOT NON AND NOT BCA
- Total   := BCA + NON BCA + NON
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype


def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_required_columns(df: pd.DataFrame) -> Dict[str, str]:
    want = {
        "amount": ["total tarif rp", "total tarif", "total tarif rupiah"],
        "paid_date": ["tanggal pembayaran", "tgl pembayaran", "tanggal bayar"],
        "origin": ["asal", "pelabuhan", "origin"],
        "pay_type": ["tipe pembayaran", "jenis pembayaran", "payment type"],
        "sof_id": ["sof id", "sof", "source of fund id", "source of funds id"],
    }
    canon_to_actual: Dict[str, str] = {_canon(c): c for c in df.columns}

    found: Dict[str, str] = {}
    for logical, candidates in want.items():
        for cand in candidates:
            if cand in canon_to_actual:
                found[logical] = canon_to_actual[cand]
                break

    missing = [k for k in want if k not in found]
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing)
            + "\n\nKolom yang ada:\n"
            + ", ".join(map(str, df.columns))
        )
    return found


def _parse_rupiah(v) -> float:
    if pd.isna(v):
        return 0.0
    s = str(v).strip()
    if not s:
        return 0.0

    s = re.sub(r"(?i)\brp\b", "", s)
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in {"-", ".", ","}:
        return 0.0

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if s.count(".") > 1 and "," not in s:
            s = s.replace(".", "")
        elif s.count(".") == 1 and "," not in s:
            left, right = s.split(".")
            if len(right) == 3:
                s = left + right

        if "," in s and "." not in s:
            left, right = s.split(",", 1)
            if len(right) in (0, 3):
                s = left + right
            else:
                s = left + "." + right

    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_payment_date(series: pd.Series) -> pd.Series:
    """
    Handles:
    - string dd/mm/yyyy hh:mm:ss
    - Excel serial date (numeric)
    """
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s = series.copy()
    as_num = pd.to_numeric(s, errors="coerce")
    num_mask = as_num.notna()

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if num_mask.any():
        out.loc[num_mask] = pd.to_datetime(
            as_num.loc[num_mask],
            unit="d",
            origin="1899-12-30",
            errors="coerce",
        )

    if (~num_mask).any():
        out.loc[~num_mask] = pd.to_datetime(
            s.loc[~num_mask],
            errors="coerce",
            dayfirst=True,
        )

    return out


@st.cache_data(show_spinner=False)
def _load_xlsb(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="pyxlsb")


def _build_rekonsiliasi_table(df_raw: pd.DataFrame, year: Optional[int], month: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = _find_required_columns(df_raw)
    df = df_raw.copy()

    df["_paid_dt"] = _parse_payment_date(df[col["paid_date"]])
    df = df[df["_paid_dt"].notna()].copy()

    if year is not None:
        df = df[df["_paid_dt"].dt.year == year]
    if month is not None:
        df = df[df["_paid_dt"].dt.month == month]

    df["_tanggal"] = df["_paid_dt"].dt.strftime("%d-%m-%Y")  # required dd-mm-yyyy
    df["_pelabuhan"] = df[col["origin"]].astype(str).fillna("").str.strip()

    tipe = df[col["pay_type"]].astype(str).fillna("").str.strip().str.lower()
    sof = df[col["sof_id"]].astype(str).fillna("").str.strip().str.lower()
    amt = df[col["amount"]].apply(_parse_rupiah)

    prepaid_any = tipe.str.contains(r"\bprepaid-", regex=True, na=False)
    cash = tipe.str.contains(r"\bcash\b", regex=True, na=False)
    skpt = tipe.str.contains("skpt", na=False)
    ifcs = tipe.str.contains("ifcs", na=False)
    reedem = tipe.str.contains(r"reedem|redeem", regex=True, na=False)

    non_bucket = cash | prepaid_any | skpt | ifcs | reedem
    bca_bucket = (~non_bucket) & sof.str.contains(r"bca|blu", regex=True, na=False)
    non_bca_bucket = (~non_bucket) & (~sof.str.contains(r"bca|blu", regex=True, na=False))

    df["NON"] = amt.where(non_bucket, 0.0)
    df["BCA"] = amt.where(bca_bucket, 0.0)
    df["NON BCA"] = amt.where(non_bca_bucket, 0.0)
    df["Total"] = df[["BCA", "NON BCA", "NON"]].sum(axis=1)

    report = (
        df.groupby(["_tanggal", "_pelabuhan"], dropna=False)[["BCA", "NON BCA", "NON", "Total"]]
        .sum()
        .reset_index()
        .rename(columns={"_tanggal": "Tanggal", "_pelabuhan": "Pelabuhan"})
        .sort_values(["Tanggal", "Pelabuhan"], ascending=[True, True])
    )

    # small debug view: date parsing + raw columns
    debug_preview = df_raw[[col["paid_date"], col["origin"], col["pay_type"], col["sof_id"], col["amount"]]].head(50).copy()
    debug_preview.insert(0, "Tanggal (parsed dd-mm-yyyy)", _parse_payment_date(df_raw[col["paid_date"]]).dt.strftime("%d-%m-%Y").head(50))

    return report, debug_preview


def main() -> None:
    st.set_page_config(page_title="Tabel Rekonsiliasi (XLSB)", layout="wide")
    st.title("Tabel Rekonsiliasi")

    uploaded = st.file_uploader("Upload Payment Report (.xlsb)", type=["xlsb"])
    if not uploaded:
        st.info("Upload file .xlsb untuk mulai.")
        return

    file_bytes = uploaded.getvalue()

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="pyxlsb")
    except ModuleNotFoundError:
        st.error("Dependency belum ada. Jalankan: pip install pyxlsb")
        return

    sheet = st.selectbox("Pilih sheet", options=xls.sheet_names, index=0)
    df_raw = _load_xlsb(file_bytes, sheet_name=sheet)

    st.sidebar.header("Parameter Rekonsiliasi (Optional)")
    now = datetime.now()
    year = st.sidebar.selectbox("Tahun", options=[None] + list(range(now.year - 5, now.year + 2)), index=0)
    month = st.sidebar.selectbox(
        "Bulan",
        options=[None] + list(range(1, 13)),
        format_func=lambda x: "Semua" if x is None else str(x),
        index=0,
    )

    st.subheader("Preview Kolom Tanggal")
    try:
        report, debug_preview = _build_rekonsiliasi_table(df_raw, year=year, month=month)
    except Exception as e:
        st.error(str(e))
        return

    st.dataframe(debug_preview, use_container_width=True)

    st.subheader("Rekonsiliasi (N–Q)")
    st.caption(f"Filter: Tahun={year if year is not None else 'Semua'}, Bulan={month if month is not None else 'Semua'}")

    if report.empty:
        st.warning("Tidak ada data setelah filter. Coba set Bulan/Tahun = Semua.")
        return

    st.dataframe(report, use_container_width=True)

    if st.checkbox("Tampilkan Grand Total"):
        totals = report[["BCA", "NON BCA", "NON", "Total"]].sum().to_frame().T
        totals.insert(0, "Pelabuhan", "ALL")
        totals.insert(0, "Tanggal", "ALL")
        st.dataframe(pd.concat([report, totals], ignore_index=True), use_container_width=True)


if __name__ == "__main__":
    main()
