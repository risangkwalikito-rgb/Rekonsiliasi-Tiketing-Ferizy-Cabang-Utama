from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class CanonicalCols:
    amount: str = "TOTAL TARIF (Rp.)"
    paid_date: str = "Tanggal Pembayaran"
    origin: str = "ASAL"
    pay_type: str = "TIPE PEMBAYARAN"
    sof_id: str = "SOF ID"


def _canonicalize_col_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_required_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Returns mapping: logical_name -> actual_df_column_name
    """
    want = {
        "amount": ["total tarif rp", "total tarif", "total tarif rupiah"],
        "paid_date": ["tanggal pembayaran", "tgl pembayaran", "tanggal bayar"],
        "origin": ["asal", "pelabuhan", "origin"],
        "pay_type": ["tipe pembayaran", "type pembayaran", "jenis pembayaran", "payment type"],
        "sof_id": ["sof id", "sof", "source of fund id", "source of funds id"],
    }

    canon_to_actual: Dict[str, str] = {}
    for col in df.columns:
        canon_to_actual[_canonicalize_col_name(col)] = col

    found: Dict[str, str] = {}
    for logical, candidates in want.items():
        for cand in candidates:
            if cand in canon_to_actual:
                found[logical] = canon_to_actual[cand]
                break

    missing = [k for k in want.keys() if k not in found]
    if missing:
        pretty = ", ".join(missing)
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {pretty}\n\nKolom yang ada di file:\n{available}"
        )
    return found


def _parse_rupiah(value) -> float:
    """
    Best-effort parsing for Indonesian currency formats:
    - "Rp 1.234.567" -> 1234567
    - "1,234,567" -> 1234567
    - "1.234.567,00" -> 1234567
    """
    if pd.isna(value):
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0

    s = re.sub(r"(?i)\brp\b", "", s)
    s = re.sub(r"[^\d,.\-]", "", s)

    if not s or s in {"-", ".", ","}:
        return 0.0

    # If both dot and comma exist, assume dot thousands + comma decimals
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # If multiple dots, treat as thousands separators
        if s.count(".") > 1 and "," not in s:
            s = s.replace(".", "")
        # If one dot and last group is 3 digits => thousands separator
        elif s.count(".") == 1 and "," not in s:
            left, right = s.split(".")
            if len(right) == 3:
                s = left + right

        # If only comma exists, usually thousands in exports; drop it
        if "," in s and "." not in s:
            left, right = s.split(",", 1)
            if len(right) in (0, 3):  # thousands or empty
                s = left + right
            else:
                s = left + "." + right

    try:
        return float(s)
    except ValueError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read_uploaded_file(file_bytes: bytes, filename: str, sheet_name: Optional[str]) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sn = sheet_name or xls.sheet_names[0]
        return pd.read_excel(xls, sheet_name=sn)
    # CSV
    return pd.read_csv(io.BytesIO(file_bytes))


def _build_report(df_raw: pd.DataFrame, year: Optional[int], month: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    colmap = _find_required_columns(df_raw)

    df = df_raw.copy()

    df["_paid_date"] = pd.to_datetime(df[colmap["paid_date"]], errors="coerce", dayfirst=True)
    df["_pelabuhan"] = df[colmap["origin"]].astype(str).fillna("").str.strip()

    df["_pay_type"] = df[colmap["pay_type"]].astype(str).fillna("").str.strip().str.lower()
    df["_sof_id"] = df[colmap["sof_id"]].astype(str).fillna("").str.strip().str.lower()

    df["_amount"] = df[colmap["amount"]].apply(_parse_rupiah)

    # Filter by month/year (rekonsiliasi)
    if year is not None:
        df = df[df["_paid_date"].dt.year == year]
    if month is not None:
        df = df[df["_paid_date"].dt.month == month]

    # Normalize date to date-only for grouping
    df["_tanggal"] = df["_paid_date"].dt.date

    # Masks (case-insensitive based on lower strings)
    tipe = df["_pay_type"]
    sof = df["_sof_id"]
    amt = df["_amount"]

    cash = tipe.str.contains(r"\bcash\b", regex=True, na=False)
    prepaid_bri = tipe.str.contains("prepaid-bri", na=False)
    prepaid_bni = tipe.str.contains("prepaid-bni", na=False)
    prepaid_mandiri = tipe.str.contains("prepaid-mandiri", na=False)
    prepaid_bca = tipe.str.contains("prepaid-bca", na=False)

    skpt = tipe.str.contains("skpt", na=False)
    ifcs = tipe.str.contains("ifcs", na=False)
    reedem = tipe.str.contains(r"reedem|redeem", regex=True, na=False)

    finpay = tipe.str.contains("finpay", na=False)
    espay = finpay & sof.str.contains("spay", na=False)
    finnet = finpay & sof.str.contains("finpay021", na=False)

    prepaid_any = tipe.str.contains(r"\bprepaid-", regex=True, na=False)

    # Rekonsiliasi buckets (mutually exclusive)
    non_bucket = cash | prepaid_any | reedem | skpt | ifcs
    bca_bucket = (~non_bucket) & sof.str.contains(r"bca|blu", regex=True, na=False)
    non_bca_bucket = (~non_bucket) & (~sof.str.contains(r"bca|blu", regex=True, na=False))

    # Create amount columns
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

    a_to_l_cols = [
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
    df["Total (A-L)"] = df[a_to_l_cols].sum(axis=1)

    df["BCA"] = amt.where(bca_bucket, 0.0)
    df["NON BCA"] = amt.where(non_bca_bucket, 0.0)
    df["NON"] = amt.where(non_bucket, 0.0)
    df["Total (N+O+P)"] = df[["BCA", "NON BCA", "NON"]].sum(axis=1)

    group_keys = ["_tanggal", "_pelabuhan"]
    out_cols = [
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
        "Total (A-L)",
        "BCA",
        "NON BCA",
        "NON",
        "Total (N+O+P)",
    ]

    report = (
        df.groupby(group_keys, dropna=False)[out_cols]
        .sum()
        .reset_index()
        .rename(columns={"_tanggal": "Tanggal", "_pelabuhan": "Pelabuhan"})
        .sort_values(["Tanggal", "Pelabuhan"], ascending=[True, True])
    )

    return report, colmap


def _to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Detail Payment Report") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Detail Payment Report", layout="wide")
    st.title("Detail Payment Report")

    st.sidebar.header("Parameter Rekonsiliasi")
    this_year = datetime.now().year
    year = st.sidebar.selectbox("Tahun", options=[None] + list(range(this_year - 5, this_year + 1)), index=1)
    month_options = [None] + list(range(1, 13))
    month = st.sidebar.selectbox("Bulan", options=month_options, format_func=lambda x: "Semua" if x is None else str(x))

    uploaded = st.file_uploader("Upload Payment Report (Excel/CSV)", type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.info("Upload file untuk mulai.")
        return

    file_bytes = uploaded.getvalue()
    filename = uploaded.name

    sheet_name = None
    if filename.lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        if len(xls.sheet_names) > 1:
            sheet_name = st.selectbox("Pilih sheet", options=xls.sheet_names)
        else:
            sheet_name = xls.sheet_names[0]

    try:
        df_raw = _read_uploaded_file(file_bytes, filename, sheet_name)
        report, colmap = _build_report(df_raw, year=year, month=month)
    except Exception as e:
        st.error(str(e))
        return

    st.subheader("Detail Payment Report (Title of The Table)")
    st.caption(
        f"Filter Rekonsiliasi: Tahun={year if year is not None else 'Semua'}, Bulan={month if month is not None else 'Semua'}"
    )

    if report.empty:
        st.warning("Tidak ada data setelah filter.")
        return

    st.dataframe(report, use_container_width=True)

    # Optional totals row
    if st.checkbox("Tampilkan Grand Total"):
        numeric_cols = report.select_dtypes(include="number").columns.tolist()
        grand = report[numeric_cols].sum().to_frame().T
        grand.insert(0, "Pelabuhan", "ALL")
        grand.insert(0, "Tanggal", "ALL")
        st.dataframe(pd.concat([report, grand], ignore_index=True), use_container_width=True)

    excel_bytes = _to_excel_bytes(report)
    st.download_button(
        "Download Excel",
        data=excel_bytes,
        file_name=f"detail_payment_report_{year or 'all'}_{month or 'all'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("Debug: Kolom yang terdeteksi"):
        st.json(colmap)


if __name__ == "__main__":
    main()
