from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype


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

    # If both dot and comma exist => dot thousands, comma decimals
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # Multiple dots => thousands
        if s.count(".") > 1 and "," not in s:
            s = s.replace(".", "")
        # One dot, last group 3 digits => thousands
        elif s.count(".") == 1 and "," not in s:
            left, right = s.split(".")
            if len(right) == 3:
                s = left + right

        # Only comma => either thousands or decimals
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
    Robust parser for XLSB:
    - If already datetime -> keep
    - If numeric -> treat as Excel serial date (days since 1899-12-30)
    - Else -> parse as string date (dayfirst)
    """
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s = series.copy()

    # Mixed types: handle numeric cells separately
    num_mask = pd.to_numeric(s, errors="coerce").notna()

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if num_mask.any():
        s_num = pd.to_numeric(s[num_mask], errors="coerce")

        # Heuristic: Excel serial dates are usually in range ~ 20000..60000
        # Still convert all numeric; non-date numeric will become NaT if too weird after conversion checks below.
        dt_num = pd.to_datetime(s_num, unit="d", origin="1899-12-30", errors="coerce")
        out.loc[num_mask] = dt_num

    if (~num_mask).any():
        dt_str = pd.to_datetime(s[~num_mask], errors="coerce", dayfirst=True)
        out.loc[~num_mask] = dt_str

    return out


@st.cache_data(show_spinner=False)
def _load_xlsb(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="pyxlsb")


def _build_report(df_raw: pd.DataFrame, year: Optional[int], month: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    col = _find_required_columns(df_raw)
    df = df_raw.copy()

    df["_paid_date"] = _parse_payment_date(df[col["paid_date"]])
    df["_tanggal"] = df["_paid_date"].dt.date
    df["_pelabuhan"] = df[col["origin"]].astype(str).fillna("").str.strip()

    df["_pay_type"] = df[col["pay_type"]].astype(str).fillna("").str.strip().str.lower()
    df["_sof_id"] = df[col["sof_id"]].astype(str).fillna("").str.strip().str.lower()

    df["_amount"] = df[col["amount"]].apply(_parse_rupiah)

    # Keep only rows where date parsed, otherwise year/month filter will drop silently
    df_valid_date = df[df["_paid_date"].notna()].copy()

    if year is not None:
        df_valid_date = df_valid_date[df_valid_date["_paid_date"].dt.year == year]
    if month is not None:
        df_valid_date = df_valid_date[df_valid_date["_paid_date"].dt.month == month]

    tipe = df_valid_date["_pay_type"]
    sof = df_valid_date["_sof_id"]
    amt = df_valid_date["_amount"]

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

    df_valid_date["Cash"] = amt.where(cash, 0.0)
    df_valid_date["Prepaid BRI"] = amt.where(prepaid_bri, 0.0)
    df_valid_date["Prepaid BNI"] = amt.where(prepaid_bni, 0.0)
    df_valid_date["Prepaid Mandiri"] = amt.where(prepaid_mandiri, 0.0)
    df_valid_date["Prepaid BCA"] = amt.where(prepaid_bca, 0.0)
    df_valid_date["SKPT"] = amt.where(skpt, 0.0)
    df_valid_date["IFCS"] = amt.where(ifcs, 0.0)
    df_valid_date["Reedem"] = amt.where(reedem, 0.0)
    df_valid_date["ESPAY"] = amt.where(espay, 0.0)
    df_valid_date["FINNET"] = amt.where(finnet, 0.0)

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
    df_valid_date["Total (A-L)"] = df_valid_date[a_to_l].sum(axis=1)

    df_valid_date["BCA"] = amt.where(bca_bucket, 0.0)
    df_valid_date["NON BCA"] = amt.where(non_bca_bucket, 0.0)
    df_valid_date["NON"] = amt.where(non_bucket, 0.0)
    df_valid_date["Total (N+O+P)"] = df_valid_date[["BCA", "NON BCA", "NON"]].sum(axis=1)

    out_cols = a_to_l + ["Total (A-L)", "BCA", "NON BCA", "NON", "Total (N+O+P)"]

    report = (
        df_valid_date.groupby(["_tanggal", "_pelabuhan"], dropna=False)[out_cols]
        .sum()
        .reset_index()
        .rename(columns={"_tanggal": "Tanggal", "_pelabuhan": "Pelabuhan"})
        .sort_values(["Tanggal", "Pelabuhan"], ascending=[True, True])
    )

    return report, col, df


def main() -> None:
    st.set_page_config(page_title="Detail Payment Report (XLSB)", layout="wide")
    st.title("Detail Payment Report")

    st.sidebar.header("Parameter Rekonsiliasi")
    now = datetime.now()
    year = st.sidebar.selectbox("Tahun", options=[None] + list(range(now.year - 5, now.year + 2)), index=1)
    month = st.sidebar.selectbox(
        "Bulan",
        options=[None] + list(range(1, 13)),
        format_func=lambda x: "Semua" if x is None else str(x),
    )

    uploaded = st.file_uploader("Upload Payment Report (.xlsb)", type=["xlsb"])
    if not uploaded:
        st.info("Upload file .xlsb untuk mulai.")
        return

    file_bytes = uploaded.getvalue()

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="pyxlsb")
        sheet = st.selectbox("Pilih sheet", options=xls.sheet_names, index=0)
        df_raw = _load_xlsb(file_bytes, sheet_name=sheet)

        report, colmap, df_debug = _build_report(df_raw, year=year, month=month)
    except ModuleNotFoundError:
        st.error("Dependency belum ada. Jalankan: pip install pyxlsb")
        return
    except Exception as e:
        st.error(str(e))
        return

    st.subheader("Detail Payment Report (Title of The Table)")
    st.caption(f"Filter: Tahun={year if year is not None else 'Semua'}, Bulan={month if month is not None else 'Semua'}")

    # Debug info: bantu jelasin kenapa kosong
    with st.expander("Debug (cek parsing tanggal & filter)"):
        st.write("Kolom terdeteksi:", colmap)

        paid = _parse_payment_date(df_raw[colmap["paid_date"]])
        st.write("Total rows:", len(df_raw))
        st.write("Tanggal ter-parse (notna):", int(paid.notna().sum()))
        if paid.notna().any():
            st.write("Min tanggal:", paid.min())
            st.write("Max tanggal:", paid.max())
            st.write("Sample 10 tanggal:", paid.dropna().head(10).tolist())

            ym = paid.dropna().dt.to_period("M").astype(str)
            st.write("Top Year-Month:", ym.value_counts().head(12))

        # show first rows raw
        st.dataframe(df_raw.head(20), use_container_width=True)

    if report.empty:
        st.warning("Tidak ada data setelah filter. Cek Debug: kemungkinan tanggal tidak terbaca sebagai 2025-10.")
        return

    st.dataframe(report, use_container_width=True)


if __name__ == "__main__":
    main()
