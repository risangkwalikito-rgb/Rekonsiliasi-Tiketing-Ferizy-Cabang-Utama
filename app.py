from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class LogicalCols:
    amount: str = "TOTAL TARIF (Rp.)"
    paid_date: str = "Tanggal Pembayaran"
    origin: str = "ASAL"
    pay_type: str = "TIPE PEMBAYARAN"
    sof_id: str = "SOF ID"


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
        s = s.replace(".", "")
        s = s.replace(",", ".")
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


@st.cache_data(show_spinner=False)
def _load_xlsb(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="pyxlsb")


def _build_report(df_raw: pd.DataFrame, year: Optional[int], month: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    col = _find_required_columns(df_raw)
    df = df_raw.copy()

    df["_paid_date"] = pd.to_datetime(df[col["paid_date"]], errors="coerce", dayfirst=True)
    df["_tanggal"] = df["_paid_date"].dt.date
    df["_pelabuhan"] = df[col["origin"]].astype(str).fillna("").str.strip()

    df["_pay_type"] = df[col["pay_type"]].astype(str).fillna("").str.strip().str.lower()
    df["_sof_id"] = df[col["sof_id"]].astype(str).fillna("").str.strip().str.lower()
    df["_amount"] = df[col["amount"]].apply(_parse_rupiah)

    if year is not None:
        df = df[df["_paid_date"].dt.year == year]
    if month is not None:
        df = df[df["_paid_date"].dt.month == month]

    tipe = df["_pay_type"]
    sof = df["_sof_id"]
    amt = df["_amount"]

    cash = tipe.str.contains(r"\bcash\b", regex=True, na=False)
    prepaid_bri = tipe.str.contains("prepaid-bri", na=False)
    prepaid_bni = tipe.str.contains("prepaid-bni", na=False)
    prepaid_mandiri = tipe.str.contains("prepaid-mandiri", na=False)
    prepaid_bca = tipe.str.contains("prepaid-bca", na=False)
    prepaid_any = tipe.str.contains(r"\bprepaid-", regex=True, na=False)

    skpt = tipe.str.contains("skpt", na=False)
    ifcs = tipe.str.contains("ifcs", na=False)
    reedem = tipe.str.contains(r"reedem|redeem", regex=True, na=False)

    finpay = tipe.str.contains("finpay", na=False)
    espay = finpay & sof.str.contains("spay", na=False)
    finnet = finpay & sof.str.contains("finpay021", na=False)

    non_bucket = cash | prepaid_any | reedem | skpt | ifcs
    bca_bucket = (~non_bucket) & sof.str.contains(r"bca|blu", regex=True, na=False)
    non_bca_bucket = (~non_bucket) & (~sof.str.contains(r"bca|blu", regex=True, na=False))

    df["Cash"] = amt.where(cash, 0.0)
    df["Prepaid BRI"] = amt.where(prepaid_bri, 0.0)
    df["Prepaid BNI"] = amt.where(prepaid_bni, 0.0)
    df["Prepaid Mandiri"] = amt.where(prepaid_mandiri, 0.0)
    df["Prepaid BCA"] = amt.where(prepaid_bca, 0.0)
    df["SKPT"] = amt.where(skpt, 0.0)
    df["IFCS"] = amt.where(ifcs, 0.0)
    df["Reedem"] = amt.where(reedem, 0.0)
    df["ESPAY"] = amt.where(espay, 0.0)
    df["FINNET"] = amt.where(finnet, 0.0)

    a_to_l = [
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
    df["Total (A-L)"] = df[a_to_l].sum(axis=1)

    df["BCA"] = amt.where(bca_bucket, 0.0)
    df["NON BCA"] = amt.where(non_bca_bucket, 0.0)
    df["NON"] = amt.where(non_bucket, 0.0)
    df["Total (N+O+P)"] = df[["BCA", "NON BCA", "NON"]].sum(axis=1)

    out_cols = a_to_l + ["Total (A-L)", "BCA", "NON BCA", "NON", "Total (N+O+P)"]

    report = (
        df.groupby(["_tanggal", "_pelabuhan"], dropna=False)[out_cols]
        .sum()
        .reset_index()
        .rename(columns={"_tanggal": "Tanggal", "_pelabuhan": "Pelabuhan"})
        .sort_values(["Tanggal", "Pelabuhan"], ascending=[True, True])
    )

    return report, col


def main() -> None:
    st.set_page_config(page_title="Detail Payment Report (XLSB)", layout="wide")
    st.title("Detail Payment Report")

    st.sidebar.header("Parameter Rekonsiliasi")
    now = datetime.now()
    year = st.sidebar.selectbox("Tahun", options=[None] + list(range(now.year - 5, now.year + 1)), index=1)
    month = st.sidebar.selectbox("Bulan", options=[None] + list(range(1, 13)), format_func=lambda x: "Semua" if x is None else str(x))

    uploaded = st.file_uploader("Upload Payment Report (.xlsb)", type=["xlsb"])
    if not uploaded:
        st.info("Upload file .xlsb untuk mulai.")
        return

    file_bytes = uploaded.getvalue()

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="pyxlsb")
        sheet = st.selectbox("Pilih sheet", options=xls.sheet_names, index=0)
        df_raw = _load_xlsb(file_bytes, sheet_name=sheet)
        report, colmap = _build_report(df_raw, year=year, month=month)
    except ModuleNotFoundError:
        st.error("Dependency belum ada. Jalankan: pip install pyxlsb")
        return
    except Exception as e:
        st.error(str(e))
        return

    st.subheader("Detail Payment Report (Title of The Table)")
    st.caption(f"Filter: Tahun={year if year is not None else 'Semua'}, Bulan={month if month is not None else 'Semua'}")

    if report.empty:
        st.warning("Tidak ada data setelah filter.")
        return

    st.dataframe(report, use_container_width=True)

    with st.expander("Debug: kolom terdeteksi"):
        st.json(colmap)


if __name__ == "__main__":
    main()
