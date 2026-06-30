"""
📚 Vocab App — Anki CSV Generator (Google Sheets edition)
Tabs: Add | Vocabulary | Generate
Card: Front = Phrase (vocab bolded orange)
      Back  = Phrase + dashed hr + POS + IPA + Indonesian sentence translation
Output: Tab-separated .txt for Anki basic import
"""

import streamlit as st
import pandas as pd
import json, re, time, io, math
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Vocab App", layout="centered", page_icon="📚")

# ─── Secrets ─────────────────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    SPREADSHEET_ID = st.secrets["SPREADSHEET_ID"]
    _GCP_INFO      = st.secrets["gcp_service_account"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

# ─── Google Sheets ────────────────────────────────────────────────────────────
@st.cache_resource
def get_sheet():
    creds = Credentials.from_service_account_info(
        dict(_GCP_INFO),
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)
    return client.open_by_key(SPREADSHEET_ID).worksheet("vocabulary")


@st.cache_data(ttl=300)
def load_vocab() -> pd.DataFrame:
    try:
        sheet = get_sheet()
        rows  = sheet.get_all_records(default_blank="")
    except Exception as e:
        st.error(f"Could not load sheet: {e}")
        st.stop()

    if not rows:
        return pd.DataFrame(columns=["vocab", "phrase", "status"])

    df = pd.DataFrame(rows).astype(str)
    for col, default in [("vocab", ""), ("phrase", ""), ("status", "New")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).replace("nan", default)

    df = (
        df[["vocab", "phrase", "status"]]
        .pipe(lambda d: d[d["vocab"].str.strip().str.len() > 0])
        .sort_values("vocab", ignore_index=True)
    )
    return df


def save_vocab(df: pd.DataFrame):
    clean = (
        df[["vocab", "phrase", "status"]]
        .copy()
        .pipe(lambda d: d[d["vocab"].astype(str).str.strip().str.len() > 0])
        .drop_duplicates("vocab", keep="last")
        .fillna("")
    )
    sheet = get_sheet()
    sheet.clear()
    data = [clean.columns.tolist()] + clean.astype(str).values.tolist()
    sheet.update(data, value_input_option="RAW")
    load_vocab.clear()


# ─── Gemini ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    )

gemini = get_gemini()

# ─── AI Prompt ───────────────────────────────────────────────────────────────
_PROMPT = """Kamu adalah kamus dwibahasa Inggris-Indonesia. Untuk setiap item di bawah,
kembalikan JSON array (urutan & jumlah sama dengan input).

KONTEKS PENTING:
- Jika field "phrase" pada input TIDAK kosong, gunakan kalimat itu sebagai KONTEKS untuk menentukan arti vocab yang paling tepat (vocab bisa punya banyak arti, pilih yang sesuai konteks kalimat).
- Jika field "phrase" pada input KOSONG, buat sendiri 1 kalimat Inggris pendek yang menunjukkan makna vocab dengan jelas.

FORMAT OUTPUT:
[
  {{
    "vocab": "sama seperti input",
    "part_of_speech": "Noun / Verb / Adjective / Adverb",
    "pronunciation_ipa": "/notasi IPA/",
    "translation": "terjemahan vocab ke Bahasa Indonesia (1-3 kata), sesuai konteks kalimat",
    "phrase_en": "Salin SAMA PERSIS dari field phrase di input jika tidak kosong. Jika kosong, kalimat buatan sendiri.",
    "phrase_id": "Terjemahan phrase_en ke Bahasa Indonesia. WAJIB gunakan PERSIS kata/frasa yang sama dengan nilai 'translation' di dalam kalimat ini, jangan diubah bentuknya (jangan ditambah imbuhan berbeda)."
  }}
]

ATURAN WAJIB:
- translation : HANYA kata/frasa terjemahan vocab (1-3 kata), bukan kalimat
- phrase_en   : salin dari input jika ada, buat sendiri jika kosong
- phrase_id   : terjemahan kalimat utuh, dan kata "translation" HARUS muncul verbatim (sama persis) di dalam phrase_id
- JANGAN tambahkan tag HTML apapun (jangan pakai <b> atau tag lain) di phrase_id maupun translation
- Output HANYA array JSON, tanpa teks tambahan apapun

INPUT:
{batch}"""



def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def generate_cards(vocab_phrase_list: list, batch_size: int = 10) -> list:
    all_data = []
    batches  = [vocab_phrase_list[i : i + batch_size] for i in range(0, len(vocab_phrase_list), batch_size)]
    prog     = st.progress(0.0)
    status   = st.empty()

    for idx, batch in enumerate(batches):
        words = ", ".join(v[0] for v in batch)
        status.info(f"⏳ Batch {idx + 1} / {len(batches)} — {words}")

        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt      = _PROMPT.format(batch=json.dumps(batch_dicts, ensure_ascii=False))
        success     = False

        for attempt in range(3):
            try:
                resp   = gemini.generate_content(prompt)
                parsed = _parse_json(resp.text)
                if isinstance(parsed, list) and parsed:
                    phrase_map = {item["vocab"]: item["phrase"] for item in batch_dicts}
                    for item in parsed:
                        original_phrase = phrase_map.get(item.get("vocab", ""), "")
                        # Pakai phrase dari input jika ada, fallback ke phrase_en buatan Gemini
                        item["phrase"] = original_phrase if original_phrase else item.get("phrase_en", "")
                    all_data.extend(parsed)
                    success = True
                    break
            except Exception as exc:
                wait = 15 + attempt * 5 if "429" in str(exc) else 2
                time.sleep(wait)

        if not success:
            for item in batch_dicts:
                all_data.append({
                    "vocab": item["vocab"],
                    "phrase": item["phrase"],
                    "part_of_speech": "",
                    "pronunciation_ipa": "",
                    "phrase_id": "",
                })

        prog.progress((idx + 1) / len(batches))
        if idx < len(batches) - 1:
            time.sleep(1)

    prog.empty()
    status.empty()
    return all_data


# ─── Build HTML card sides ────────────────────────────────────────────────────
def _bold_vocab(phrase: str, vocab: str) -> str:
    """Wrap vocab word in phrase with <b> tag only, no style."""
    return re.sub(
        rf'(?i)\b({re.escape(vocab)})\b',
        r'<b>\1</b>',
        phrase,
    )


def build_front(note: dict) -> str:
    vocab  = note.get("vocab", "").strip()
    phrase = note.get("phrase", "").strip()
    # Only bold the vocab word inside the sentence
    if phrase:
        return _bold_vocab(phrase, vocab)
    return f"<b>{vocab}</b>"


def _bold_translation(phrase_id: str, translation: str) -> str:
    """Bold the translation word(s) inside the Indonesian sentence.
    Tries exact word-boundary match first, falls back to plain substring
    match (case-insensitive) if word-boundary match fails (e.g. due to
    Indonesian affixes)."""
    if not phrase_id or not translation:
        return phrase_id

    # Strip any HTML the model might have added anyway
    clean_phrase = re.sub(r"</?b>", "", phrase_id)
    clean_translation = re.sub(r"</?b>", "", translation).strip()

    # 1) Try exact word-boundary match
    pattern = rf'(?i)\b({re.escape(clean_translation)})\b'
    new_phrase, count = re.subn(pattern, r'<b>\1</b>', clean_phrase, count=1)
    if count > 0:
        return new_phrase

    # 2) Fallback: plain case-insensitive substring match (no boundaries)
    idx = clean_phrase.lower().find(clean_translation.lower())
    if idx != -1:
        end = idx + len(clean_translation)
        return clean_phrase[:idx] + f"<b>{clean_phrase[idx:end]}</b>" + clean_phrase[end:]

    # 3) Last resort: try bolding just the first word of translation
    first_word = clean_translation.split()[0] if clean_translation.split() else ""
    if first_word:
        pattern2 = rf'(?i)\b({re.escape(first_word)})\b'
        new_phrase, count = re.subn(pattern2, r'<b>\1</b>', clean_phrase, count=1)
        if count > 0:
            return new_phrase

    return clean_phrase  # no match found, return unbolded


def build_back(note: dict) -> str:
    pos         = note.get("part_of_speech", "").strip()
    ipa         = note.get("pronunciation_ipa", "").strip()
    translation = note.get("translation", "").strip()
    phrase_id   = note.get("phrase_id", "").strip()

    # Bold POS, plain IPA
    if pos and ipa:
        meta = f"<b>{pos}</b>. {ipa}"
    elif pos:
        meta = f"<b>{pos}</b>"
    else:
        meta = ipa

    phrase_id_html = _bold_translation(phrase_id, translation)

    parts = []
    if meta:
        parts.append(meta)
    if phrase_id_html:
        parts.append(f"<br>{phrase_id_html}")

    return "<br>".join(parts)


# ─── Create Anki tab-separated CSV ───────────────────────────────────────────
def create_anki_csv(notes: list) -> bytes:
    lines = ["#separator:tab", "#html:true", "#columns:Front\tBack"]
    for note in notes:
        front = build_front(note).replace("\n", "").replace("\t", " ")
        back  = build_back(note).replace("\n", "").replace("\t", " ")
        lines.append(f"{front}\t{back}")
    return "\n".join(lines).encode("utf-8")


# ─── Session state ────────────────────────────────────────────────────────────
st.session_state.setdefault("vocab_df",      load_vocab().copy())
st.session_state.setdefault("csv_bytes",     None)
st.session_state.setdefault("preview_notes", [])

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📚 Vocab App")

df      = st.session_state.vocab_df
total   = len(df)
new_ct  = int((df["status"] == "New").sum())
done_ct = int((df["status"] == "Done").sum())

m1, m2, m3 = st.columns(3)
m1.metric("Total",  total)
m2.metric("New ✨",  new_ct)
m3.metric("Done ✅", done_ct)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_add, tab_vocab, tab_gen = st.tabs([
    "➕ Add",
    f"📖 Vocabulary ({total})",
    f"📇 Generate ({new_ct} New)",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — ADD
# ══════════════════════════════════════════════════════════════
with tab_add:
    st.subheader("Add a word")

    v_in = st.text_input("Word", placeholder="e.g. submersible", key="t1_vocab")
    p_in = st.text_input(
        "Example sentence (optional)",
        placeholder="A submersible can explore the deep ocean floor.",
        key="t1_phrase",
    )

    if st.button("💾 Save", type="primary", use_container_width=True):
        v = v_in.strip().lower()
        p = p_in.strip()
        if not v:
            st.error("Please enter a word.")
        elif (st.session_state.vocab_df["vocab"] == v).any():
            st.warning(f"'{v}' already exists.")
        else:
            row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New"}])
            st.session_state.vocab_df = pd.concat(
                [st.session_state.vocab_df, row], ignore_index=True
            )
            save_vocab(st.session_state.vocab_df)
            st.success(f"✅ '{v}' saved!")
            st.rerun()

    st.divider()
    st.subheader("Bulk add")
    st.caption("Satu kata per baris. Atau: `kata, kalimat contoh`")

    bulk_text = st.text_area(
        "Words",
        height=150,
        placeholder="apple\nsubmersible, A submersible can explore the deep ocean floor.\nephemeral",
        key="t1_bulk",
        label_visibility="collapsed",
    )

    if st.button("➕ Add All", use_container_width=True):
        added, skipped, new_rows = 0, 0, []
        for line in bulk_text.strip().splitlines():
            parts = line.strip().split(",", 1)
            v = parts[0].strip().lower()
            p = parts[1].strip() if len(parts) > 1 else ""
            if not v:
                continue
            if (st.session_state.vocab_df["vocab"] == v).any():
                skipped += 1
                continue
            new_rows.append({"vocab": v, "phrase": p, "status": "New"})
            added += 1

        if added:
            st.session_state.vocab_df = pd.concat(
                [st.session_state.vocab_df, pd.DataFrame(new_rows)], ignore_index=True
            )
            save_vocab(st.session_state.vocab_df)
            msg = f"✅ Added {added} word(s)."
            if skipped:
                msg += f" Skipped {skipped} duplicate(s)."
            st.success(msg)
            st.rerun()
        elif skipped:
            st.warning(f"All {skipped} word(s) already exist.")
        else:
            st.warning("No valid words found in the input.")

# ══════════════════════════════════════════════════════════════
#  TAB 2 — VOCABULARY
# ══════════════════════════════════════════════════════════════
with tab_vocab:
    st.subheader(f"Vocabulary ({total} words)")

    search  = st.text_input("🔎 Search", "", key="t2_search").strip().lower()
    show_df = st.session_state.vocab_df.copy()
    if search:
        show_df = show_df[show_df["vocab"].str.contains(search, case=False, na=False)]

    edited_df = st.data_editor(
        show_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status", options=["New", "Done"], required=True
            )
        },
    )

    col_save, col_reset = st.columns(2)

    if col_save.button("💾 Save Changes", type="primary", use_container_width=True):
        full = st.session_state.vocab_df.copy()
        common = [i for i in edited_df.index if i in full.index]
        if common:
            full.loc[common, ["vocab", "phrase", "status"]] = (
                edited_df.loc[common, ["vocab", "phrase", "status"]].values
            )
        new_idx = [i for i in edited_df.index if i not in full.index]
        if new_idx:
            full = pd.concat(
                [full, edited_df.loc[new_idx, ["vocab", "phrase", "status"]]],
                ignore_index=True,
            )
        removed = [i for i in show_df.index if i not in edited_df.index]
        if removed:
            full = full.drop(index=removed).reset_index(drop=True)

        st.session_state.vocab_df = full
        save_vocab(full)
        st.toast("✅ Saved to Google Sheets!")
        st.rerun()

    if col_reset.button("🔄 Reset All to New", use_container_width=True):
        st.session_state.vocab_df["status"] = "New"
        save_vocab(st.session_state.vocab_df)
        st.toast("🔄 All reset to New!")
        st.rerun()

    st.divider()
    st.download_button(
        "💾 Download backup CSV",
        data=st.session_state.vocab_df.to_csv(index=False).encode("utf-8"),
        file_name=f"vocab_backup_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════
#  TAB 3 — GENERATE
# ══════════════════════════════════════════════════════════════
with tab_gen:

    if st.session_state.csv_bytes is not None:
        st.success(f"✅ {len(st.session_state.preview_notes)} cards ready!")

        st.download_button(
            "📥 Download Anki CSV (.txt)",
            data=st.session_state.csv_bytes,
            file_name=f"vocab_anki_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption("Import di Anki: **File → Import** → pilih file ini → separator = **Tab**, centang **Allow HTML**")

        if st.session_state.preview_notes:
            with st.expander("👁️ Preview kartu (5 pertama)", expanded=True):
                for note in st.session_state.preview_notes[:5]:
                    vocab     = note.get("vocab", "?")
                    phrase    = note.get("phrase", "")
                    pos       = note.get("part_of_speech", "")
                    ipa       = note.get("pronunciation_ipa", "")
                    phrase_id = note.get("phrase_id", "—")

                    st.markdown(f"**Front:** {phrase or vocab}")
                    st.markdown(f"**Back:** *(kalimat sama)* → `{pos}` {ipa}  \n🇮🇩 _{phrase_id}_")
                    st.divider()

        if st.button("🔄 Generate lagi", use_container_width=True):
            st.session_state.csv_bytes     = None
            st.session_state.preview_notes = []
            st.rerun()

    else:
        new_df = st.session_state.vocab_df[st.session_state.vocab_df["status"] == "New"].copy()

        if new_df.empty:
            st.warning(
                "Tidak ada kata 'New'. "
                "Tambah di tab **Add**, atau reset status di **Vocabulary**."
            )
        else:
            st.subheader("Generate Anki CSV")

            st.write("**Pilih kata yang mau di-export:**")
            sel_df = new_df[["vocab", "phrase"]].copy()
            sel_df.insert(0, "Export", True)

            edited_sel = st.data_editor(
                sel_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Export": st.column_config.CheckboxColumn("Export?", required=True),
                    "vocab":  st.column_config.TextColumn("Word",   disabled=True),
                    "phrase": st.column_config.TextColumn("Phrase", disabled=True),
                },
            )

            selected = edited_sel[edited_sel["Export"] == True]
            n        = len(selected)
            n_batch  = math.ceil(n / 10) if n > 0 else 0

            if n > 0:
                st.info(f"📦 **{n}** kata  ·  **{n_batch}** batch  ·  ~{n_batch * 5}–{n_batch * 8}s")

                if st.button("🚀 Generate Cards", type="primary", use_container_width=True):
                    vocab_list = selected[["vocab", "phrase"]].values.tolist()
                    notes = generate_cards(vocab_list, batch_size=10)

                    if notes:
                        csv_data = create_anki_csv(notes)

                        done_vocabs = {n["vocab"] for n in notes}
                        st.session_state.vocab_df.loc[
                            st.session_state.vocab_df["vocab"].isin(done_vocabs), "status"
                        ] = "Done"
                        save_vocab(st.session_state.vocab_df)

                        st.session_state.csv_bytes     = csv_data
                        st.session_state.preview_notes = notes
                        st.rerun()
                    else:
                        st.error("❌ Gagal generate. Cek Gemini API key dan quota.")
            else:
                st.warning("Pilih minimal satu kata.")
