"""
📚 Vocab App — Lightweight Anki Generator
Tabs: Add | Vocabulary | Generate
Card: Front = Word  |  Back = Translation (ID) + Definition (ID) + IPA + Synonym/Antonym
Theme: Minimalistic (white, Inter font, indigo accents)
Audio: disabled
Batch size: 10 words / request
"""

import streamlit as st
import pandas as pd
import io, json, re, time, os, tempfile, hashlib, math
import google.generativeai as genai
import genanki
from datetime import datetime

try:
    from github import Github, GithubException
except ImportError:
    st.error("Install PyGithub: `pip install PyGithub`")
    st.stop()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Vocab App", layout="centered", page_icon="📚")

# ─── Secrets ─────────────────────────────────────────────────────────────────
try:
    GITHUB_TOKEN   = st.secrets["GITHUB_TOKEN"]
    REPO_NAME      = st.secrets["REPO_NAME"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Check your .streamlit/secrets.toml")
    st.stop()

# ─── GitHub ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_repo():
    return Github(GITHUB_TOKEN).get_repo(REPO_NAME)

repo = get_repo()

# ─── Gemini ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    )

gemini = get_gemini()

# ─── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_vocab() -> pd.DataFrame:
    try:
        f  = repo.get_contents("vocabulary.csv")
        df = pd.read_csv(io.StringIO(f.decoded_content.decode("utf-8")), dtype=str)
    except GithubException as e:
        if e.status == 404:
            return pd.DataFrame(columns=["vocab", "phrase", "status"])
        st.error(f"GitHub error: {e}")
        st.stop()
    for col, default in [("phrase", ""), ("status", "New")]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    return df[["vocab", "phrase", "status"]].sort_values("vocab", ignore_index=True)


def save_vocab(df: pd.DataFrame):
    clean = (
        df[["vocab", "phrase", "status"]]
        .copy()
        .pipe(lambda d: d[d["vocab"].astype(str).str.strip().str.len() > 0])
        .drop_duplicates("vocab", keep="last")
    )
    csv = clean.to_csv(index=False)
    try:
        f = repo.get_contents("vocabulary.csv")
        repo.update_file(f.path, "Update vocab", csv, f.sha)
    except GithubException as e:
        if e.status == 404:
            repo.create_file("vocabulary.csv", "Init vocab", csv)
        else:
            raise
    load_vocab.clear()


# ─── AI generation ────────────────────────────────────────────────────────────
PROMPT = """\
Kamu adalah kamus dwibahasa Inggris-Indonesia. Untuk setiap kata Inggris di bawah,
kembalikan JSON array (urutan & jumlah sama dengan input).

FORMAT OUTPUT:
[
  {{
    "vocab": "sama seperti input",
    "translation": "terjemahan Indonesia, 1-3 kata saja (bukan kalimat)",
    "definition_id": "Definisi singkat bahasa Indonesia, maks 12 kata.",
    "part_of_speech": "Noun / Verb / Adjective / Adverb",
    "pronunciation_ipa": "/notasi IPA/",
    "synonym": "satu sinonim Inggris yang paling umum",
    "antonym": "satu antonim Inggris yang paling umum, atau string kosong jika tidak ada"
  }}
]

ATURAN WAJIB:
- translation  : HANYA kata/frasa Indonesia, BUKAN kalimat penuh
- definition_id: definisi pendek Bahasa Indonesia, maks 12 kata
- synonym      : TEPAT satu sinonim saja
- antonym      : tepat satu antonim, atau "" jika memang tidak ada
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
        status.info(f"⏳ Batch {idx + 1} / {len(batches)} — {', '.join(v[0] for v in batch)}")
        batch_dicts = [{"vocab": v[0], "phrase": v[1]} for v in batch]
        prompt      = PROMPT.format(batch=json.dumps(batch_dicts, ensure_ascii=False))
        success     = False

        for attempt in range(3):
            try:
                resp   = gemini.generate_content(prompt)
                parsed = _parse_json(resp.text)
                if isinstance(parsed, list) and parsed:
                    all_data.extend(parsed)
                    success = True
                    break
            except Exception as exc:
                wait = 15 + attempt * 5 if "429" in str(exc) else 2
                time.sleep(wait)

        if not success:
            for item in batch_dicts:
                all_data.append({
                    "vocab": item["vocab"], "translation": "—",
                    "definition_id": "", "part_of_speech": "",
                    "pronunciation_ipa": "", "synonym": "", "antonym": "",
                })

        prog.progress((idx + 1) / len(batches))
        if idx < len(batches) - 1:
            time.sleep(1)  # gentle rate-limit buffer

    prog.empty()
    status.empty()
    return all_data


# ─── Anki card theme (Minimalistic) ──────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
.card {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #ffffff;
    color: #1e293b;
    padding: 36px 28px 32px;
    text-align: center;
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto;
}
.word {
    font-size: 2.5em;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin-bottom: 6px;
}
.pos {
    font-size: 0.68em;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
hr {
    border: none;
    border-top: 1.5px solid #f1f5f9;
    margin: 22px 0;
}
.translation {
    font-size: 1.9em;
    font-weight: 700;
    color: #4f46e5;
    margin-bottom: 6px;
}
.definition {
    font-size: 0.9em;
    color: #64748b;
    font-style: italic;
    margin-bottom: 10px;
}
.ipa {
    font-family: 'Courier New', monospace;
    font-size: 0.85em;
    color: #94a3b8;
}
.syn-row {
    display: flex;
    justify-content: center;
    gap: 48px;
    flex-wrap: wrap;
    margin-top: 2px;
}
.pill-label {
    font-size: 0.6em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
    margin-bottom: 3px;
}
.pill-val {
    font-size: 0.88em;
    color: #475569;
}
"""

_FRONT = """
<div class="card">
  <div class="word">{{Word}}</div>
  {{#PartOfSpeech}}<div class="pos">{{PartOfSpeech}}</div>{{/PartOfSpeech}}
</div>
"""

_BACK = """
<div class="card">
  <div class="word">{{Word}}</div>
  {{#PartOfSpeech}}<div class="pos">{{PartOfSpeech}}</div>{{/PartOfSpeech}}
  <hr>
  <div class="translation">{{Translation}}</div>
  {{#DefinitionID}}<div class="definition">{{DefinitionID}}</div>{{/DefinitionID}}
  {{#Pronunciation}}<div class="ipa">{{Pronunciation}}</div>{{/Pronunciation}}
  {{#Synonym}}
  <hr>
  <div class="syn-row">
    <div>
      <div class="pill-label">Sinonim</div>
      <div class="pill-val">{{Synonym}}</div>
    </div>
    {{#Antonym}}
    <div>
      <div class="pill-label">Antonim</div>
      <div class="pill-val">{{Antonym}}</div>
    </div>
    {{/Antonym}}
  </div>
  {{/Synonym}}
</div>
"""


def create_apkg(notes: list, deck_name: str, deck_id: int) -> bytes:
    model_id = int(hashlib.md5(("MinimalVocab_v1_" + deck_name).encode()).hexdigest(), 16) % (1 << 31)

    my_model = genanki.Model(
        model_id,
        "Minimal Vocab v1",
        fields=[
            {"name": "Word"},
            {"name": "Translation"},
            {"name": "DefinitionID"},
            {"name": "Pronunciation"},
            {"name": "PartOfSpeech"},
            {"name": "Synonym"},
            {"name": "Antonym"},
        ],
        templates=[{"name": "Recognition", "qfmt": _FRONT, "afmt": _BACK}],
        css=_CSS,
    )

    my_deck = genanki.Deck(deck_id, deck_name)

    for note in notes:
        vocab = str(note.get("vocab", "")).strip()
        if not vocab:
            continue
        guid = str(int(hashlib.sha256((vocab + deck_name).encode()).hexdigest(), 16) % (10 ** 10))
        my_deck.add_note(genanki.Note(
            model=my_model,
            guid=guid,
            fields=[
                vocab,
                str(note.get("translation", "")),
                str(note.get("definition_id", "")),
                str(note.get("pronunciation_ipa", "")),
                str(note.get("part_of_speech", "")),
                str(note.get("synonym", "")),
                str(note.get("antonym", "")),
            ],
        ))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "deck.apkg")
        genanki.Package(my_deck).write_to_file(path)
        with open(path, "rb") as f:
            return f.read()


# ─── Session state ────────────────────────────────────────────────────────────
st.session_state.setdefault("vocab_df",      load_vocab().copy())
st.session_state.setdefault("apkg_bytes",    None)
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

    v_in = st.text_input("Word", placeholder="e.g. serendipity", key="t1_vocab")
    p_in = st.text_input(
        "Example sentence (optional)",
        placeholder="She found the café by serendipity.",
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
    st.caption("One word per line. Or: `word, example sentence`")

    bulk_text = st.text_area(
        "Words",
        height=150,
        placeholder="apple\nserendipity, She found it by serendipity.\nephemeral",
        key="t1_bulk",
        label_visibility="collapsed",
    )

    if st.button("➕ Add All", use_container_width=True):
        added, skipped = 0, 0
        for line in bulk_text.strip().splitlines():
            parts = line.strip().split(",", 1)
            v = parts[0].strip().lower()
            p = parts[1].strip() if len(parts) > 1 else ""
            if not v:
                continue
            if (st.session_state.vocab_df["vocab"] == v).any():
                skipped += 1
                continue
            row = pd.DataFrame([{"vocab": v, "phrase": p, "status": "New"}])
            st.session_state.vocab_df = pd.concat(
                [st.session_state.vocab_df, row], ignore_index=True
            )
            added += 1

        if added:
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
        # Update rows that were shown
        common = [i for i in edited_df.index if i in full.index]
        if common:
            full.loc[common, ["vocab", "phrase", "status"]] = (
                edited_df.loc[common, ["vocab", "phrase", "status"]].values
            )
        # New rows added inside the editor
        new_idx = [i for i in edited_df.index if i not in full.index]
        if new_idx:
            full = pd.concat([full, edited_df.loc[new_idx, ["vocab", "phrase", "status"]]], ignore_index=True)
        # Deleted rows
        removed = [i for i in show_df.index if i not in edited_df.index]
        if removed:
            full = full.drop(index=removed).reset_index(drop=True)
        st.session_state.vocab_df = full
        save_vocab(full)
        st.toast("✅ Saved!")
        st.rerun()

    if col_reset.button("🔄 Reset All to New", use_container_width=True):
        st.session_state.vocab_df["status"] = "New"
        save_vocab(st.session_state.vocab_df)
        st.toast("🔄 All reset to New!")
        st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 3 — GENERATE
# ══════════════════════════════════════════════════════════════
with tab_gen:

    # ── If deck is ready, show download ──────────────────────
    if st.session_state.apkg_bytes is not None:
        st.success(f"✅ Deck ready — {len(st.session_state.preview_notes)} cards!")

        st.download_button(
            "📥 Download .apkg",
            data=st.session_state.apkg_bytes,
            file_name=f"vocab_{datetime.now().strftime('%Y%m%d_%H%M')}.apkg",
            mime="application/octet-stream",
            use_container_width=True,
        )

        if st.session_state.preview_notes:
            with st.expander("👁️ Card preview (first 5)", expanded=True):
                for note in st.session_state.preview_notes[:5]:
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(
                            f"**{note.get('vocab','?')}**  \n"
                            f"_{note.get('pronunciation_ipa','')}_  \n"
                            f"*{note.get('part_of_speech','')}*"
                        )
                    with c2:
                        st.markdown(
                            f"🇮🇩 **{note.get('translation','—')}**  \n"
                            f"{note.get('definition_id','')}  \n"
                            f"Syn: _{note.get('synonym','—')}_ &nbsp;·&nbsp; Ant: _{note.get('antonym','—') or '—'}_"
                        )
                    st.divider()

        if st.button("🔄 Generate another deck", use_container_width=True):
            st.session_state.apkg_bytes    = None
            st.session_state.preview_notes = []
            st.rerun()

    # ── Otherwise show generation form ───────────────────────
    else:
        new_df = st.session_state.vocab_df[st.session_state.vocab_df["status"] == "New"].copy()

        if new_df.empty:
            st.warning("No 'New' words to generate. Add words in the **Add** tab, or reset statuses in **Vocabulary**.")
        else:
            st.subheader("Generate Anki Deck")
            deck_name = st.text_input(
                "Deck name (use :: for sub-decks)",
                value="English::Vocabulary",
                key="t3_deck",
            )

            st.write("**Select words to export:**")
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
                st.info(
                    f"📦 **{n}** words  ·  **{n_batch}** batch(es) of 10  ·  "
                    f"~{n_batch * 5}–{n_batch * 8}s estimate"
                )

                if st.button("🚀 Generate Cards", type="primary", use_container_width=True):
                    vocab_list = selected[["vocab", "phrase"]].values.tolist()
                    notes = generate_cards(vocab_list, batch_size=10)

                    if notes:
                        clean_name = deck_name.strip() or "Vocabulary"
                        deck_id    = (
                            int(hashlib.sha256(clean_name.encode()).hexdigest(), 16) % (1 << 30)
                            + (1 << 29)
                        )
                        with st.spinner("📦 Packing .apkg…"):
                            apkg_data = create_apkg(notes, clean_name, deck_id)

                        # Mark generated words as Done
                        done_vocabs = {n["vocab"] for n in notes}
                        st.session_state.vocab_df.loc[
                            st.session_state.vocab_df["vocab"].isin(done_vocabs), "status"
                        ] = "Done"
                        save_vocab(st.session_state.vocab_df)

                        st.session_state.apkg_bytes    = apkg_data
                        st.session_state.preview_notes = notes
                        st.rerun()
                    else:
                        st.error("❌ Generation failed. Check your Gemini API key and quota.")
            else:
                st.warning("Select at least one word.")

