from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Rekonsiliasi Otomatis (Payment vs Settlement vs Rekening Koran)"
CACHE_DIR = Path(".rekon_cache_uploads")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PD_DAYFIRST_DEFAULT = True


# -----------------------------
# Helpers: normalization
# -----------------------------
def _norm_col_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("_", " ")
    return s


def _build_col_map(columns: Sequence[str]) -> dict[str, str]:
    """
    Map normalized name -> original name (first occurrence).
    """
    out: dict[str, str] = {}
    for c in columns:
        nc = _norm_col_name(c)
        if nc not in out:
            out[nc] = c
    return out


def _pick_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """
    Pick first matching column (case-insensitive, space/underscore tolerant).
    """
    col_map = _build_col_map(columns)
    for cand in candidates:
        key = _norm_col_name(cand)
        if key in col_map:
            return col_map[key]
    return None


def _require_cols(df: pd.DataFrame, required_candidates: dict[str, Sequence[str]], ctx: str) -> dict[str, str]:
    """
    Resolve required columns by candidates; raise user-friendly error if missing.
    Returns alias -> actual_col.
    """
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for alias, cands in required_candidates.items():
        col = _pick_col(df.columns.tolist(), list(cands))
        if not col:
            missing.append(f"{alias} (candidates: {', '.join(cands)})")
        else:
            resolved[alias] = col

    if missing:
        raise ValueError(f"[{ctx}] Kolom wajib tidak ditemukan:\n- " + "\n- ".join(missing))

    return resolved


def _contains(series: pd.Series, needle: str) -> pd.Series:
    s = series.astype("string").fillna("")
    return s.str.contains(needle, case=False, na=False, regex=False)


# -----------------------------
# Helpers: parsing numbers/dates
# -----------------------------
def _clean_amount_series(s: pd.Series) -> pd.Series:
    """
    Robust parsing for IDR-like numbers:
    - Removes currency and spaces.
    - Handles thousand separators (.) and decimal separators (,).
    - Handles parentheses as negative.
    Returns float64.
    """
    x = s.astype("string").fillna("").str.strip()

    neg = x.str.match(r"^\(.*\)$")
    x = x.str.replace(r"[()]", "", regex=True)

    # remove currency symbols and non-numeric separators, keep digits, dot, comma, minus
    x = x.str.replace(r"[^0-9,\.\-]", "", regex=True)

    def _normalize_one(val: str) -> str:
        if not val:
            return ""
        # If both '.' and ',', decide decimal by last separator
        if "." in val and "," in val:
            if val.rfind(".") > val.rfind(","):
                # dot is decimal: remove commas (thousand)
                val = val.replace(",", "")
            else:
                # comma is decimal: remove dots (thousand), replace comma -> dot
                val = val.replace(".", "").replace(",", ".")
            return val

        # Only comma
        if "," in val and "." not in val:
            if val.count(",") > 1:
                # assume commas are thousand separators
                return val.replace(",", "")
            # single comma: decide decimal if 1-2 digits after
            parts = val.split(",")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
                return parts[0].replace(",", "") + "." + parts[1]
            return val.replace(",", "")

        # Only dot
        if "." in val and "," not in val:
            if val.count(".") > 1:
                # assume dot thousand separators
                return val.replace(".", "")
            parts = val.split(".")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
                return val
            return val.replace(".", "")

        return val

    normalized = x.map(_normalize_one)
    out = pd.to_numeric(normalized, errors="coerce").fillna(0.0)
    out = out.where(~neg, -out)
    return out.astype("float64")


def _to_date_series(s: pd.Series, dayfirst: bool = PD_DAYFIRST_DEFAULT) -> pd.Series:
    dtv = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return dtv.dt.normalize()


def _period_range(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(dt.date(year, month, 1))
    if month == 12:
        end = pd.Timestamp(dt.date(year + 1, 1, 1))
    else:
        end = pd.Timestamp(dt.date(year, month + 1, 1))
    return start, end


def _filter_period(df: pd.DataFrame, date_col: str, year: int, month: int) -> pd.DataFrame:
    start, end = _period_range(year, month)
    return df[(df[date_col] >= start) & (df[date_col] < end)]


# -----------------------------
# Helpers: upload caching to disk
# -----------------------------
@dataclass(frozen=True)
class SavedUpload:
    name: str
    path: Path
    md5: str


def _md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()


def _save_uploaded_files(files: Sequence[st.runtime.uploaded_file_manager.UploadedFile]) -> list[SavedUpload]:
    saved: list[SavedUpload] = []
    for f in files:
        b = f.getvalue()
        h = _md5_bytes(b)
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f.name)
        out_path = CACHE_DIR / f"{h}_{safe_name}"
        if not out_path.exists():
            out_path.write_bytes(b)
        saved.append(SavedUpload(name=f.name, path=out_path, md5=h))
    return saved


def _iter_zip_members(zip_path: Path) -> Iterable[Tuple[str, bytes]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, "r") as fh:
                yield info.filename, fh.read()


# -----------------------------
# Readers (header-aware usecols)
# -----------------------------
def _read_excel_header(path_or_buf, engine: str | None) -> list[str]:
    df0 = pd.read_excel(path_or_buf, engine=engine, nrows=0)
    return df0.columns.tolist()


def _read_csv_header(path_or_buf) -> list[str]:
    df0 = pd.read_csv(path_or_buf, nrows=0)
    return df0.columns.tolist()


def _read_excel_mincols(path_or_buf, engine: str | None, candidates: Sequence[str]) -> pd.DataFrame:
    cols = _read_excel_header(path_or_buf, engine=engine)
    picked = []
    for c in candidates:
        pc = _pick_col(cols, [c])
        if pc and pc not in picked:
            picked.append(pc)
    if not picked:
        # fallback: read all (last resort)
        return pd.read_excel(path_or_buf, engine=engine)
    return pd.read_excel(path_or_buf, engine=engine, usecols=picked)


def _read_csv_mincols(path_or_buf, candidates: Sequence[str], chunksize: Optional[int] = None):
    cols = _read_csv_header(path_or_buf)
    picked = []
    for c in candidates:
        pc = _pick_col(cols, [c])
        if pc and pc not in picked:
            picked.append(pc)
    if not picked:
        picked = None  # type: ignore[assignment]
    return pd.read_csv(path_or_buf, usecols=picked, chunksize=chunksize)


def _infer_engine_from_ext(ext: str) -> Optional[str]:
    ext = ext.lower()
    if ext == ".xlsb":
        return "pyxlsb"
    if ext == ".xlsx":
        return "openpyxl"
    if ext == ".xls":
        return "xlrd"
    return None


# -----------------------------
# Processing: Payment Report
# -----------------------------
PAYMENT_COLS = [
    "Tanggal Pembayaran",
    "ASAL",
    "TIPE PEMBAYARAN",
    "TOTAL TARIF (Rp.)",
    "SOF ID",
    "nomer invoice",
]


@st.cache_data(show_spinner=False)
def process_payment_report(paths: Tuple[str, ...], year: int, month: int, dayfirst: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for p in paths:
        path = Path(p)
        ext = path.suffix.lower()

        if ext == ".zip":
            for member_name, member_bytes in _iter_zip_members(path):
                mext = Path(member_name).suffix.lower()
                if mext in (".xlsb", ".xlsx", ".xls"):
                    engine = _infer_engine_from_ext(mext)
                    bio = io.BytesIO(member_bytes)
                    df = _read_excel_mincols(bio, engine=engine, candidates=PAYMENT_COLS)
                    frames.append(df)
        elif ext in (".xlsb", ".xlsx", ".xls"):
            engine = _infer_engine_from_ext(ext)
            df = _read_excel_mincols(path, engine=engine, candidates=PAYMENT_COLS)
            frames.append(df)
        else:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    cols = _require_cols(
        df,
        {
            "tanggal": ["Tanggal Pembayaran"],
            "pelabuhan": ["ASAL"],
            "tipe": ["TIPE PEMBAYARAN"],
            "amount": ["TOTAL TARIF (Rp.)"],
            "sof": ["SOF ID"],
            "invoice": ["nomer invoice"],
        },
        ctx="Payment Report",
    )

    out = pd.DataFrame(
        {
            "Tanggal": _to_date_series(df[cols["tanggal"]], dayfirst=dayfirst),
            "Pelabuhan": df[cols["pelabuhan"]].astype("string").fillna("").str.strip(),
            "Tipe": df[cols["tipe"]].astype("string").fillna("").str.strip(),
            "SOF": df[cols["sof"]].astype("string").fillna("").str.strip(),
            "Invoice": df[cols["invoice"]].astype("string").fillna("").str.strip(),
            "Amount": _clean_amount_series(df[cols["amount"]]),
        }
    )

    out = out[out["Tanggal"].notna()]
    out = _filter_period(out, "Tanggal", year, month)

    # normalize for matching
    tipe_l = out["Tipe"].astype("string").str.lower().fillna("")
    sof_l = out["SOF"].astype("string").str.lower().fillna("")

    is_cash = _contains(tipe_l, "cash")
    is_pre_bri = _contains(tipe_l, "prepaid-bri")
    is_pre_bni = _contains(tipe_l, "prepaid-bni")
    is_pre_mandiri = _contains(tipe_l, "prepaid-mandiri")
    is_pre_bca = _contains(tipe_l, "prepaid-bca")
    is_skpt = _contains(tipe_l, "skpt")
    is_ifcs = _contains(tipe_l, "ifcs")
    is_redeem = _contains(tipe_l, "reedem") | _contains(tipe_l, "redeem")

    is_finpay = _contains(tipe_l, "finpay")
    is_espay = is_finpay & _contains(sof_l, "spay")
    is_finnet = is_finpay & _contains(sof_l, "finpay021")

    is_bca = _contains(sof_l, "bca") | _contains(sof_l, "blu")
    is_non_bca = ~is_bca

    is_non_bucket = (
        is_cash
        | is_pre_bri
        | is_pre_bni
        | is_pre_mandiri
        | is_pre_bca
        | is_skpt
        | is_ifcs
        | is_redeem
    )

    group_cols = ["Tanggal", "Pelabuhan"]

    def _sum(mask: pd.Series) -> pd.Series:
        return out.loc[mask].groupby(group_cols)["Amount"].sum()

    def _nunique(mask: pd.Series) -> pd.Series:
        return out.loc[mask].groupby(group_cols)["Invoice"].nunique()

    # Amount buckets (Detail Payment)
    s_cash = _sum(is_cash)
    s_pre_bri = _sum(is_pre_bri)
    s_pre_bni = _sum(is_pre_bni)
    s_pre_mandiri = _sum(is_pre_mandiri)
    s_pre_bca = _sum(is_pre_bca)
    s_skpt = _sum(is_skpt)
    s_ifcs = _sum(is_ifcs)
    s_redeem = _sum(is_redeem)
    s_espay = _sum(is_espay)
    s_finnet = _sum(is_finnet)

    s_total_al = (
        s_cash
        .add(s_pre_bri, fill_value=0)
        .add(s_pre_bni, fill_value=0)
        .add(s_pre_mandiri, fill_value=0)
        .add(s_pre_bca, fill_value=0)
        .add(s_skpt, fill_value=0)
        .add(s_ifcs, fill_value=0)
        .add(s_redeem, fill_value=0)
        .add(s_espay, fill_value=0)
        .add(s_finnet, fill_value=0)
    )

    s_bca = _sum(is_bca)
    s_nonbca = _sum(is_non_bca)
    s_non = _sum(is_non_bucket)
    s_total_np = s_bca.add(s_nonbca, fill_value=0).add(s_non, fill_value=0)

    # Ticket detail for recon (ESPAY vs FINNET)
    is_spay = _contains(sof_l, "spay")
    is_not_spay = ~is_spay

    s_ticket_espay_bca = _sum(is_spay & is_bca)
    s_ticket_espay_nonbca = _sum(is_spay & is_non_bca)
    n_ticket_espay_bca = _nunique(is_spay & is_bca)
    n_ticket_espay_nonbca = _nunique(is_spay & is_non_bca)

    s_ticket_finnet_bca = _sum(is_not_spay & is_bca)
    s_ticket_finnet_nonbca = _sum(is_not_spay & is_non_bca)
    n_ticket_finnet_bca = _nunique(is_not_spay & is_bca)
    n_ticket_finnet_nonbca = _nunique(is_not_spay & is_non_bca)

    idx = (
        out[group_cols]
        .drop_duplicates()
        .set_index(group_cols)
        .index
        .union(s_total_al.index)
        .union(s_total_np.index)
        .union(s_ticket_espay_bca.index)
        .union(s_ticket_finnet_bca.index)
    )

    def _series_to_col(s: pd.Series, name: str) -> pd.Series:
        return s.reindex(idx).fillna(0.0).rename(name)

    def _series_to_int(s: pd.Series, name: str) -> pd.Series:
        return s.reindex(idx).fillna(0).astype("int64").rename(name)

    detail = pd.concat(
        [
            _series_to_col(s_cash, "Cash"),
            _series_to_col(s_pre_bri, "Prepaid BRI"),
            _series_to_col(s_pre_bni, "Prepaid BNI"),
            _series_to_col(s_pre_mandiri, "Prepaid Mandiri"),
            _series_to_col(s_pre_bca, "Prepaid BCA"),
            _series_to_col(s_skpt, "SKPT"),
            _series_to_col(s_ifcs, "IFCS"),
            _series_to_col(s_redeem, "Reedem"),
            _series_to_col(s_espay, "ESPAY"),
            _series_to_col(s_finnet, "FINNET"),
            _series_to_col(s_total_al, "Total (A-L)"),
            _series_to_col(s_bca, "BCA (SOF)"),
            _series_to_col(s_nonbca, "NON BCA (SOF)"),
            _series_to_col(s_non, "NON (Tipe)"),
            _series_to_col(s_total_np, "Total (N-P)"),
            # For reconciliation
            _series_to_col(s_ticket_espay_bca, "Tiket ESPAY BCA (Amt)"),
            _series_to_col(s_ticket_espay_nonbca, "Tiket ESPAY NON BCA (Amt)"),
            _series_to_int(n_ticket_espay_bca, "Tiket ESPAY BCA (Cnt)"),
            _series_to_int(n_ticket_espay_nonbca, "Tiket ESPAY NON BCA (Cnt)"),
            _series_to_col(s_ticket_finnet_bca, "Tiket FINNET BCA (Amt)"),
            _series_to_col(s_ticket_finnet_nonbca, "Tiket FINNET NON BCA (Amt)"),
            _series_to_int(n_ticket_finnet_bca, "Tiket FINNET BCA (Cnt)"),
            _series_to_int(n_ticket_finnet_nonbca, "Tiket FINNET NON BCA (Cnt)"),
        ],
        axis=1,
    ).reset_index()

    detail.sort_values(["Tanggal", "Pelabuhan"], inplace=True)
    return detail


# -----------------------------
# Processing: Settlement ESPAY
# -----------------------------
ESPAY_COLS = [
    "Date",
    "Merchant_Name",
    "VA Name",
    "Channel",
    "Bank",
    "Product Name",
    "Amount",
    "Tx Fee",
    "Settlement Amount",
]


@st.cache_data(show_spinner=False)
def process_settlement_espay(paths: Tuple[str, ...], year: int, month: int, dayfirst: bool) -> pd.DataFrame:
    aggs: list[pd.DataFrame] = []

    group_cols = ["Tanggal", "Pelabuhan"]

    def _process_df(df: pd.DataFrame) -> pd.DataFrame:
        cols_date = _pick_col(df.columns.tolist(), ["Date"])
        cols_pel = _pick_col(df.columns.tolist(), ["Merchant_Name", "VA Name"])
        cols_channel = _pick_col(df.columns.tolist(), ["Channel"])
        cols_bank = _pick_col(df.columns.tolist(), ["Bank"])
        cols_prod = _pick_col(df.columns.tolist(), ["Product Name"])
        cols_settle = _pick_col(df.columns.tolist(), ["Settlement Amount"])
        cols_amt = _pick_col(df.columns.tolist(), ["Amount"])
        cols_fee = _pick_col(df.columns.tolist(), ["Tx Fee"])

        if not cols_date or not cols_pel or not cols_channel:
            raise ValueError("[Settlement ESPAY] minimal butuh kolom: Date, (Merchant_Name/VA Name), Channel")

        tanggal = _to_date_series(df[cols_date], dayfirst=dayfirst)
        pel = df[cols_pel].astype("string").fillna("").str.strip()
        channel = df[cols_channel].astype("string").fillna("")
        bank = df[cols_bank].astype("string").fillna("") if cols_bank else pd.Series([""] * len(df))
        prod = df[cols_prod].astype("string").fillna("") if cols_prod else pd.Series([""] * len(df))

        if cols_settle:
            amount = _clean_amount_series(df[cols_settle])
        else:
            if not cols_amt:
                raise ValueError("[Settlement ESPAY] kolom Amount atau Settlement Amount tidak ditemukan")
            amt = _clean_amount_series(df[cols_amt])
            fee = _clean_amount_series(df[cols_fee]) if cols_fee else 0.0
            amount = amt - fee

        base = pd.DataFrame(
            {
                "Tanggal": tanggal,
                "Pelabuhan": pel,
                "Channel": channel,
                "Bank": bank,
                "Product": prod,
                "Amount": amount,
            }
        )
        base = base[base["Tanggal"].notna()]
        base = _filter_period(base, "Tanggal", year, month)

        ch_l = base["Channel"].astype("string").str.lower().fillna("")
        bca_flag = (
            _contains(base["Bank"].astype("string").str.lower(), "bca")
            | _contains(base["Product"].astype("string").str.lower(), "bca")
            | _contains(ch_l, "bca")
        )
        va_flag = _contains(ch_l, "va")

        def _g(mask: pd.Series, colname: str) -> pd.DataFrame:
            g = base.loc[mask].groupby(group_cols).agg(
                **{f"{colname} (Amt)": ("Amount", "sum"), f"{colname} (Cnt)": ("Amount", "size")}
            )
            return g

        g_va = _g(va_flag, "VA")
        g_em = _g(~va_flag, "E-Money")
        g_bca = _g(bca_flag, "BCA")
        g_non = _g(~bca_flag, "NON BCA")

        g_all = g_va.join(g_em, how="outer").join(g_bca, how="outer").join(g_non, how="outer").fillna(0)
        g_all["Total VA+E (Amt)"] = g_all["VA (Amt)"] + g_all["E-Money (Amt)"]
        g_all["Total VA+E (Cnt)"] = g_all["VA (Cnt)"] + g_all["E-Money (Cnt)"]
        g_all["Total BCA+NON (Amt)"] = g_all["BCA (Amt)"] + g_all["NON BCA (Amt)"]
        g_all["Total BCA+NON (Cnt)"] = g_all["BCA (Cnt)"] + g_all["NON BCA (Cnt)"]
        return g_all.reset_index()

    for p in paths:
        path = Path(p)
        ext = path.suffix.lower()

        if ext == ".zip":
            for member_name, member_bytes in _iter_zip_members(path):
                mext = Path(member_name).suffix.lower()
                if mext == ".csv":
                    bio = io.BytesIO(member_bytes)
                    # chunking for speed/memory
                    reader = _read_csv_mincols(bio, ESPAY_COLS, chunksize=400_000)
                    acc: Optional[pd.DataFrame] = None
                    for chunk in reader:
                        gg = _process_df(chunk).set_index(group_cols)
                        acc = gg if acc is None else acc.add(gg, fill_value=0)
                    if acc is not None:
                        aggs.append(acc.reset_index())
                elif mext in (".xlsx", ".xls"):
                    engine = _infer_engine_from_ext(mext)
                    bio = io.BytesIO(member_bytes)
                    df = _read_excel_mincols(bio, engine=engine, candidates=ESPAY_COLS)
                    aggs.append(_process_df(df))
        elif ext == ".csv":
            reader = _read_csv_mincols(path, ESPAY_COLS, chunksize=400_000)
            acc2: Optional[pd.DataFrame] = None
            for chunk in reader:
                gg = _process_df(chunk).set_index(group_cols)
                acc2 = gg if acc2 is None else acc2.add(gg, fill_value=0)
            if acc2 is not None:
                aggs.append(acc2.reset_index())
        elif ext in (".xlsx", ".xls"):
            engine = _infer_engine_from_ext(ext)
            df = _read_excel_mincols(path, engine=engine, candidates=ESPAY_COLS)
            aggs.append(_process_df(df))

    if not aggs:
        return pd.DataFrame()

    out = pd.concat(aggs, ignore_index=True)
    out = out.groupby(group_cols, as_index=False).sum(numeric_only=True)
    out.sort_values(["Tanggal", "Pelabuhan"], inplace=True)
    return out


# -----------------------------
# Processing: Settlement FINNET
# -----------------------------
FINNET_COLS = [
    "Payment Date Time",
    "Merchant Name",
    "Payment Method",
    "Merchant Amount",
]


@st.cache_data(show_spinner=False)
def process_settlement_finnet(paths: Tuple[str, ...], year: int, month: int, dayfirst: bool) -> pd.DataFrame:
    aggs: list[pd.DataFrame] = []
    group_cols = ["Tanggal", "Pelabuhan"]

    def _process_df(df: pd.DataFrame) -> pd.DataFrame:
        cols = _require_cols(
            df,
            {
                "tanggal": ["Payment Date Time"],
                "pelabuhan": ["Merchant Name"],
                "method": ["Payment Method"],
                "amount": ["Merchant Amount"],
            },
            ctx="Settlement FINNET",
        )
        base = pd.DataFrame(
            {
                "Tanggal": _to_date_series(df[cols["tanggal"]], dayfirst=dayfirst),
                "Pelabuhan": df[cols["pelabuhan"]].astype("string").fillna("").str.strip(),
                "Method": df[cols["method"]].astype("string").fillna(""),
                "Amount": _clean_amount_series(df[cols["amount"]]),
            }
        )
        base = base[base["Tanggal"].notna()]
        base = _filter_period(base, "Tanggal", year, month)

        m_l = base["Method"].astype("string").str.lower().fillna("")
        va_flag = _contains(m_l, "va")
        bca_flag = _contains(m_l, "bca") | _contains(m_l, "blu")

        def _g(mask: pd.Series, colname: str) -> pd.DataFrame:
            g = base.loc[mask].groupby(group_cols).agg(
                **{f"{colname} (Amt)": ("Amount", "sum"), f"{colname} (Cnt)": ("Amount", "size")}
            )
            return g

        g_va = _g(va_flag, "VA")
        g_em = _g(~va_flag, "E-Money")
        g_bca = _g(bca_flag, "BCA")
        g_non = _g(~bca_flag, "NON BCA")

        g_all = g_va.join(g_em, how="outer").join(g_bca, how="outer").join(g_non, how="outer").fillna(0)
        g_all["Total VA+E (Amt)"] = g_all["VA (Amt)"] + g_all["E-Money (Amt)"]
        g_all["Total VA+E (Cnt)"] = g_all["VA (Cnt)"] + g_all["E-Money (Cnt)"]
        g_all["Total BCA+NON (Amt)"] = g_all["BCA (Amt)"] + g_all["NON BCA (Amt)"]
        g_all["Total BCA+NON (Cnt)"] = g_all["BCA (Cnt)"] + g_all["NON BCA (Cnt)"]
        return g_all.reset_index()

    for p in paths:
        path = Path(p)
        ext = path.suffix.lower()

        if ext == ".zip":
            for member_name, member_bytes in _iter_zip_members(path):
                mext = Path(member_name).suffix.lower()
                if mext == ".csv":
                    bio = io.BytesIO(member_bytes)
                    reader = _read_csv_mincols(bio, FINNET_COLS, chunksize=500_000)
                    acc: Optional[pd.DataFrame] = None
                    for chunk in reader:
                        gg = _process_df(chunk).set_index(group_cols)
                        acc = gg if acc is None else acc.add(gg, fill_value=0)
                    if acc is not None:
                        aggs.append(acc.reset_index())
        elif ext == ".csv":
            reader = _read_csv_mincols(path, FINNET_COLS, chunksize=500_000)
            acc2: Optional[pd.DataFrame] = None
            for chunk in reader:
                gg = _process_df(chunk).set_index(group_cols)
                acc2 = gg if acc2 is None else acc2.add(gg, fill_value=0)
            if acc2 is not None:
                aggs.append(acc2.reset_index())

    if not aggs:
        return pd.DataFrame()

    out = pd.concat(aggs, ignore_index=True)
    out = out.groupby(group_cols, as_index=False).sum(numeric_only=True)
    out.sort_values(["Tanggal", "Pelabuhan"], inplace=True)
    return out


# -----------------------------
# Processing: Rekening Koran (Dana Masuk)
# -----------------------------
RK_DATE_CANDS = ["Tanggal", "Date", "Transaction Date", "Tanggal Transaksi"]
RK_KET_CANDS = ["Keterangan", "Remark", "Description", "Uraian"]
RK_CREDIT_CANDS = ["Kredit", "Credit", "CR", "Masuk", "Setoran", "Nominal Kredit"]
RK_AMOUNT_CANDS = ["Amount", "Nominal", "Mutasi"]


def _read_rekening_koran_one(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        engine = _infer_engine_from_ext(ext)
        return pd.read_excel(path, engine=engine)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".zip":
        frames = []
        for member_name, member_bytes in _iter_zip_members(path):
            mext = Path(member_name).suffix.lower()
            if mext == ".csv":
                frames.append(pd.read_csv(io.BytesIO(member_bytes)))
            elif mext in (".xlsx", ".xls"):
                engine = _infer_engine_from_ext(mext)
                frames.append(pd.read_excel(io.BytesIO(member_bytes), engine=engine))
        if frames:
            return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _extract_dana_masuk(
    rk_df: pd.DataFrame,
    year: int,
    month: int,
    keyword_any: Sequence[str],
    dayfirst: bool,
    shift_days_back: int,
) -> pd.DataFrame:
    if rk_df.empty:
        return pd.DataFrame(columns=["Tanggal", "Amount"])

    date_col = _pick_col(rk_df.columns.tolist(), RK_DATE_CANDS)
    ket_col = _pick_col(rk_df.columns.tolist(), RK_KET_CANDS)
    if not date_col or not ket_col:
        raise ValueError("Rekening Koran: minimal butuh kolom Tanggal/Date dan Keterangan/Remark")

    # amount: prefer Kredit/Credit, else Amount/Mutasi (ambil yang positif)
    credit_col = _pick_col(rk_df.columns.tolist(), RK_CREDIT_CANDS)
    amount_col = _pick_col(rk_df.columns.tolist(), RK_AMOUNT_CANDS)

    if credit_col:
        amt = _clean_amount_series(rk_df[credit_col])
    elif amount_col:
        tmp = _clean_amount_series(rk_df[amount_col])
        amt = tmp.where(tmp > 0, 0.0)
    else:
        raise ValueError("Rekening Koran: kolom nominal tidak ditemukan (Kredit/Credit atau Amount/Mutasi)")

    tanggal = _to_date_series(rk_df[date_col], dayfirst=dayfirst)
    ket = rk_df[ket_col].astype("string").fillna("")

    mask = pd.Series(False, index=rk_df.index)
    for kw in keyword_any:
        mask = mask | _contains(ket, kw)

    base = pd.DataFrame({"Tanggal": tanggal, "Amount": amt})
    base = base[base["Tanggal"].notna() & mask]

    # Align "days +1" => shift back 1 day so the recon date matches payment/settlement date.
    base["Tanggal"] = base["Tanggal"] - pd.Timedelta(days=shift_days_back)

    base = _filter_period(base, "Tanggal", year, month)
    base = base.groupby(["Tanggal"], as_index=False).sum(numeric_only=True)
    return base


@st.cache_data(show_spinner=False)
def process_rekening_koran(
    paths: Tuple[str, ...],
    year: int,
    month: int,
    dayfirst: bool,
    keywords: Tuple[str, ...],
    shift_days_back: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = _read_rekening_koran_one(Path(p))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    rk = pd.concat(frames, ignore_index=True)
    return _extract_dana_masuk(
        rk_df=rk,
        year=year,
        month=month,
        keyword_any=list(keywords),
        dayfirst=dayfirst,
        shift_days_back=shift_days_back,
    )


# -----------------------------
# Reconciliation builders
# -----------------------------
def _merge_recon_base(
    payment_detail: pd.DataFrame,
    settle_df: pd.DataFrame,
    rk_bca: pd.DataFrame,
    rk_non: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    mode: "ESPAY" or "FINNET"
    """
    if payment_detail.empty and settle_df.empty and rk_bca.empty and rk_non.empty:
        return pd.DataFrame()

    group_cols = ["Tanggal", "Pelabuhan"]

    # Payment ticket columns
    if mode == "ESPAY":
        p_bca_amt = "Tiket ESPAY BCA (Amt)"
        p_non_amt = "Tiket ESPAY NON BCA (Amt)"
        p_bca_cnt = "Tiket ESPAY BCA (Cnt)"
        p_non_cnt = "Tiket ESPAY NON BCA (Cnt)"
    else:
        p_bca_amt = "Tiket FINNET BCA (Amt)"
        p_non_amt = "Tiket FINNET NON BCA (Amt)"
        p_bca_cnt = "Tiket FINNET BCA (Cnt)"
        p_non_cnt = "Tiket FINNET NON BCA (Cnt)"

    pay_cols = group_cols + [p_bca_amt, p_non_amt, p_bca_cnt, p_non_cnt]
    p = payment_detail[pay_cols].copy() if not payment_detail.empty else pd.DataFrame(columns=pay_cols)

    # Settlement columns
    if settle_df.empty:
        s = pd.DataFrame(columns=group_cols + ["BCA (Amt)", "NON BCA (Amt)", "BCA (Cnt)", "NON BCA (Cnt)"])
    else:
        need = ["BCA (Amt)", "NON BCA (Amt)", "BCA (Cnt)", "NON BCA (Cnt)"]
        have = [c for c in need if c in settle_df.columns]
        s = settle_df[group_cols + have].copy()
        for c in need:
            if c not in s.columns:
                s[c] = 0

    # Bank incoming: only date-level; expand to pelabuhan via join later (match by date only)
    rk_b = rk_bca.rename(columns={"Amount": "Dana Masuk - BCA"}).copy() if not rk_bca.empty else pd.DataFrame(columns=["Tanggal", "Dana Masuk - BCA"])
    rk_n = rk_non.rename(columns={"Amount": "Dana Masuk - NON BCA"}).copy() if not rk_non.empty else pd.DataFrame(columns=["Tanggal", "Dana Masuk - NON BCA"])

    # Build base index from payment & settlement (date+pelabuhan)
    base_idx = pd.Index([])
    if not p.empty:
        base_idx = base_idx.union(pd.MultiIndex.from_frame(p[group_cols]))
    if not s.empty:
        base_idx = base_idx.union(pd.MultiIndex.from_frame(s[group_cols]))

    if len(base_idx) == 0:
        return pd.DataFrame()

    base = pd.DataFrame(index=base_idx).reset_index()
    base.columns = group_cols

    out = base.merge(p, on=group_cols, how="left").merge(s, on=group_cols, how="left")
    out = out.fillna(0)

    out["Total Tiket Detail (Amt)"] = out[p_bca_amt] + out[p_non_amt]
    out["Total Tiket Detail (Cnt)"] = out[p_bca_cnt] + out[p_non_cnt]

    out["Total Settlement Report (Amt)"] = out["BCA (Amt)"] + out["NON BCA (Amt)"]
    out["Total Settlement Report (Cnt)"] = out["BCA (Cnt)"] + out["NON BCA (Cnt)"]

    # Add bank amounts by date (same for all pelabuhan if rekening koran tidak punya pelabuhan)
    out = out.merge(rk_b, on="Tanggal", how="left").merge(rk_n, on="Tanggal", how="left").fillna(0)
    out["Total Dana Masuk"] = out["Dana Masuk - BCA"] + out["Dana Masuk - NON BCA"]

    out["Selisih Tiket Detail vs Settlement (Amt)"] = out["Total Tiket Detail (Amt)"] - out["Total Settlement Report (Amt)"]
    out["Selisih Dana Masuk vs Settlement (Amt)"] = out["Total Dana Masuk"] - out["Total Settlement Report (Amt)"]

    out.sort_values(group_cols, inplace=True)
    return out


def _build_summary(recon: pd.DataFrame, period_label: str) -> pd.DataFrame:
    if recon.empty:
        return pd.DataFrame()

    g = recon.groupby(["Pelabuhan"], as_index=False).agg(
        MenuPayment_Cnt=("Total Tiket Detail (Cnt)", "sum"),
        MenuPayment_Amt=("Total Tiket Detail (Amt)", "sum"),
        Settlement_Cnt=("Total Settlement Report (Cnt)", "sum"),
        Settlement_Amt=("Total Settlement Report (Amt)", "sum"),
        RekKoran_Amt=("Total Dana Masuk", "sum"),
    )

    g["Selisih_Menu_vs_Settlement_Cnt"] = g["MenuPayment_Cnt"] - g["Settlement_Cnt"]
    g["Selisih_Menu_vs_Settlement_Amt"] = g["MenuPayment_Amt"] - g["Settlement_Amt"]
    g["Selisih_RekKoran_vs_Settlement_Amt"] = g["RekKoran_Amt"] - g["Settlement_Amt"]

    g.insert(0, "Periode", period_label)
    g.rename(columns={"Pelabuhan": "Cabang"}, inplace=True)
    return g


# -----------------------------
# Export
# -----------------------------
def _to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = re.sub(r"[\[\]\*\?/\\:]", "_", name)[:31]
            (df if df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name=safe)
    return bio.getvalue()


# -----------------------------
# Streamlit UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("Parameter Periode")
        today = dt.date.today()
        year = st.number_input("Year", min_value=2018, max_value=2100, value=today.year, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, value=today.month, step=1)
        dayfirst = st.checkbox("Parsing tanggal day-first (DD/MM)", value=PD_DAYFIRST_DEFAULT)

        st.divider()
        st.header("Upload Files (multi)")
        up_payment = st.file_uploader("Payment Report (.xlsb/.zip)", type=["xlsb", "zip"], accept_multiple_files=True)
        up_espay = st.file_uploader("Settlement ESPAY (.csv/.xlsx/.xls)", type=["csv", "xlsx", "xls", "zip"], accept_multiple_files=True)
        up_finnet = st.file_uploader("Settlement FINNET (.zip/.csv)", type=["zip", "csv"], accept_multiple_files=True)
        up_rk_bca = st.file_uploader("Rekening Koran BCA (.xls/.xlsx/.csv)", type=["xls", "xlsx", "csv", "zip"], accept_multiple_files=True)
        up_rk_non = st.file_uploader("Rekening Koran NON BCA (.xls/.xlsx/.csv)", type=["xls", "xlsx", "csv", "zip"], accept_multiple_files=True)

        st.divider()
        run = st.button("🚀 Proses Rekonsiliasi", type="primary", use_container_width=True)

    if not run:
        st.info("Upload file → klik **Proses Rekonsiliasi**.")
        return

    # Dependency checks (why: xlsb/xls support depends on engine)
    try:
        import pyxlsb  # noqa: F401
    except Exception:
        st.warning("Engine xlsb (pyxlsb) tidak terdeteksi. Install `pyxlsb` untuk baca Payment Report .xlsb.")

    if up_payment:
        saved_payment = _save_uploaded_files(up_payment)
    else:
        saved_payment = []

    if up_espay:
        saved_espay = _save_uploaded_files(up_espay)
    else:
        saved_espay = []

    if up_finnet:
        saved_finnet = _save_uploaded_files(up_finnet)
    else:
        saved_finnet = []

    if up_rk_bca:
        saved_rk_bca = _save_uploaded_files(up_rk_bca)
    else:
        saved_rk_bca = []

    if up_rk_non:
        saved_rk_non = _save_uploaded_files(up_rk_non)
    else:
        saved_rk_non = []

    pay_paths = tuple(str(x.path) for x in saved_payment)
    esp_paths = tuple(str(x.path) for x in saved_espay)
    fin_paths = tuple(str(x.path) for x in saved_finnet)
    rk_bca_paths = tuple(str(x.path) for x in saved_rk_bca)
    rk_non_paths = tuple(str(x.path) for x in saved_rk_non)

    period_label = f"{int(month):02d}-{int(year)}"

    try:
        with st.spinner("Processing Payment Report..."):
            df_payment_detail = process_payment_report(pay_paths, int(year), int(month), bool(dayfirst))

        with st.spinner("Processing Settlement ESPAY..."):
            df_settle_espay = process_settlement_espay(esp_paths, int(year), int(month), bool(dayfirst))

        with st.spinner("Processing Settlement FINNET..."):
            df_settle_finnet = process_settlement_finnet(fin_paths, int(year), int(month), bool(dayfirst))

        with st.spinner("Processing Rekening Koran (ESPAY keywords SGW)..."):
            rk_bca_sgw = process_rekening_koran(
                rk_bca_paths, int(year), int(month), bool(dayfirst), keywords=("SGW",), shift_days_back=1
            )
            rk_non_sgw = process_rekening_koran(
                rk_non_paths, int(year), int(month), bool(dayfirst), keywords=("SGW",), shift_days_back=1
            )

        with st.spinner("Processing Rekening Koran (FINNET keywords FINIF/FINON)..."):
            rk_bca_fin = process_rekening_koran(
                rk_bca_paths, int(year), int(month), bool(dayfirst), keywords=("FINIF", "FINON"), shift_days_back=1
            )
            rk_non_fin = process_rekening_koran(
                rk_non_paths, int(year), int(month), bool(dayfirst), keywords=("FINIF", "FINON"), shift_days_back=1
            )

        with st.spinner("Building Rekonsiliasi..."):
            recon_espay = _merge_recon_base(df_payment_detail, df_settle_espay, rk_bca_sgw, rk_non_sgw, mode="ESPAY")
            recon_finnet = _merge_recon_base(df_payment_detail, df_settle_finnet, rk_bca_fin, rk_non_fin, mode="FINNET")

        with st.spinner("Building Summary..."):
            summary_espay = _build_summary(recon_espay, period_label)
            summary_finnet = _build_summary(recon_finnet, period_label)

    except Exception as e:
        st.error(str(e))
        st.stop()

    tabs = st.tabs(
        [
            "1) Detail Payment Report",
            "2) Detail Settlement ESPAY",
            "3) Detail Settlement FINNET",
            "4) Rekonsiliasi ESPAY",
            "5) Rekonsiliasi FINNET",
            "6) Summary ESPAY",
            "7) Summary FINNET",
            "⬇️ Download",
        ]
    )

    with tabs[0]:
        st.subheader("Detail Payment Report")
        st.dataframe(df_payment_detail, use_container_width=True)

    with tabs[1]:
        st.subheader("Detail Settlement ESPAY")
        st.dataframe(df_settle_espay, use_container_width=True)

    with tabs[2]:
        st.subheader("Detail Settlement FINNET")
        st.dataframe(df_settle_finnet, use_container_width=True)

    with tabs[3]:
        st.subheader("Rekonsiliasi ESPAY")
        st.dataframe(recon_espay, use_container_width=True)

    with tabs[4]:
        st.subheader("Rekonsiliasi FINNET")
        st.dataframe(recon_finnet, use_container_width=True)

    with tabs[5]:
        st.subheader("Summary ESPAY")
        st.dataframe(summary_espay, use_container_width=True)

    with tabs[6]:
        st.subheader("Summary FINNET")
        st.dataframe(summary_finnet, use_container_width=True)

    with tabs[7]:
        st.subheader("Download Excel (multi-sheet)")
        sheets = {
            "Detail_Payment": df_payment_detail,
            "Detail_Settlement_ESPAY": df_settle_espay,
            "Detail_Settlement_FINNET": df_settle_finnet,
            "Rekonsiliasi_ESPAY": recon_espay,
            "Rekonsiliasi_FINNET": recon_finnet,
            "Summary_ESPAY": summary_espay,
            "Summary_FINNET": summary_finnet,
        }
        excel_bytes = _to_excel_bytes(sheets)
        st.download_button(
            label="Download Rekonsiliasi.xlsx",
            data=excel_bytes,
            file_name=f"Rekonsiliasi_{period_label}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.caption("Catatan: Rekening Koran umumnya tidak punya kolom Pelabuhan; Dana Masuk di-join by Tanggal saja.")


if __name__ == "__main__":
    main()
