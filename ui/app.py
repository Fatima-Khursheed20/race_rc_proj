import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from sklearn.metrics import classification_report

    _SK_ANALYTICS = True
except ImportError:
    _SK_ANALYTICS = False

# Project root → import ``src.inference``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import inference

APP_TITLE = "RACE Quiz Lab"
PRIMARY = "#2563eb"
SURFACE = "#f8fafc"
CARD_BORDER = "#e2e8f0"
EVAL_SNAPSHOT_DIR = ROOT / "report" / "eval_snapshots"


def inject_styles():
    st.markdown(
        f"""
        <style>
        :root {{
            --rc-primary: {PRIMARY};
            --rc-surface: {SURFACE};
            --rc-border: {CARD_BORDER};
        }}
        .block-container {{ padding-top: 1.25rem; max-width: 1100px; }}
        h1 {{ font-weight: 700; letter-spacing: -0.02em; color: #0f172a; }}
        .rc-card {{
            background: #fff;
            border: 1px solid var(--rc-border);
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }}
        .rc-pill {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }}
        .rc-on {{ background: #dcfce7; color: #166534; }}
        .rc-off {{ background: #fee2e2; color: #991b1b; }}
        div[data-testid="stSidebar"] {{ background: var(--rc-surface); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_race_sample(csv_path: str = "data/raw/test.csv") -> Optional[Dict[str, Any]]:
    p = Path(csv_path)
    if not p.exists():
        return None
    import random

    with p.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return None
        return random.choice(reader)


def init_session():
    if "session_log" not in st.session_state:
        st.session_state.session_log = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "hints_opened" not in st.session_state:
        st.session_state.hints_opened = 0
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""
    if "force_generate" not in st.session_state:
        st.session_state.force_generate = False
    if "distractor_lambda" not in st.session_state:
        st.session_state.distractor_lambda = 0.5
    if "quiz_stem_generation" not in st.session_state:
        st.session_state.quiz_stem_generation = 0
    if "verification_ensemble" not in st.session_state:
        st.session_state.verification_ensemble = "soft_ls"
    if "use_question_generator" not in st.session_state:
        st.session_state.use_question_generator = True


def add_log(entry: Dict[str, Any]):
    st.session_state.session_log.append(entry)


def load_eval_snapshot(filename: str) -> Optional[Dict[str, Any]]:
    p = EVAL_SNAPSHOT_DIR / filename
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def export_logs_csv() -> None:
    if not st.session_state.session_log:
        st.info("No logs yet")
        return
    output = io.StringIO()
    keys = sorted({k for row in st.session_state.session_log for k in row.keys()})
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in st.session_state.session_log:
        flat = {}
        for k in keys:
            v = row.get(k)
            if v is None:
                flat[k] = ""
            elif isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        writer.writerow(flat)
    st.download_button("Download session log (CSV)", output.getvalue(), file_name="session_log.csv")


def _session_model_a_vectors(
    window: List[Dict[str, Any]],
) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for e in window:
        ci = e.get("correct_index")
        pi = e.get("predicted_correct_index")
        if ci is not None and pi is not None:
            y_true.append(int(ci))
            y_pred.append(int(pi))
    return y_true, y_pred


def screen_analytics():
    st.subheader("4 · Developer / analytics")
    inner = st.tabs(["Session (last N)", "Offline · Model B snapshot", "Session log"])

    with inner[0]:
        log = st.session_state.session_log
        n_log = len(log)
        if n_log == 0:
            st.info("Generate quizzes from **Article** to populate Model A and latency series.")
        else:
            if n_log == 1:
                N = 1
                st.caption("Showing the only session entry (last **1** inference).")
            else:
                N = st.slider(
                    "Last N inferences (quiz generations)",
                    1,
                    n_log,
                    value=min(50, n_log),
                    help="Each entry is one pipeline run (Model A + distractors + hints).",
                )
            window = log[-N:]

            st.markdown("##### Model A (predicted option vs gold)")
            y_true, y_pred = _session_model_a_vectors(window)
            if not y_true:
                st.caption("No rows with both `correct_index` and `predicted_correct_index`.")
            elif not _SK_ANALYTICS:
                st.warning("Install `scikit-learn` for the classification report.")
            else:
                labels = [0, 1, 2, 3]
                names = ["A", "B", "C", "D"]
                rep = classification_report(
                    y_true,
                    y_pred,
                    labels=labels,
                    target_names=names,
                    zero_division=0,
                )
                st.code(rep, language="text")

            answered = [e for e in window if e.get("was_correct") is not None]
            if answered:
                acc_u = sum(1 for e in answered if e["was_correct"]) / len(answered)
                st.metric("Learner accuracy (submitted checks, this window)", f"{acc_u:.1%}")
            else:
                st.caption("Use **Check answer** on the Quiz tab to record learner correctness.")

            st.markdown("##### Latency (ms per pipeline run)")
            lat_rows = [
                {"i": i + 1, "latency_ms": e["latency_ms"]}
                for i, e in enumerate(window)
                if e.get("latency_ms") is not None
            ]
            if lat_rows:
                s = pd.Series([r["latency_ms"] for r in lat_rows])
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean", f"{s.mean():.0f} ms")
                c2.metric("p50", f"{s.median():.0f} ms")
                c3.metric("Max", f"{s.max():.0f} ms")
                st.line_chart(pd.DataFrame(lat_rows).set_index("i")["latency_ms"])
            else:
                st.caption("No latency values in this window.")

    with inner[1]:
        snap = load_eval_snapshot("model_b_distractor.json")
        if not snap:
            st.warning(
                f"Missing `{EVAL_SNAPSHOT_DIR / 'model_b_distractor.json'}`. "
                "Add a snapshot (e.g. from saved confusion matrices) for the dashboard."
            )
        else:
            st.caption(snap.get("source", ""))
            for split in ("dev", "test"):
                block = snap.get(split)
                if not isinstance(block, dict):
                    continue
                st.markdown(f"**{split.upper()}** · ranker binary metrics (from confusion matrix)")
                m = block
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                c2.metric("Precision (distractor)", f"{m.get('precision_distractor', 0):.4f}")
                c3.metric("Recall (distractor)", f"{m.get('recall_distractor', 0):.4f}")
                c4.metric("F1 (distractor)", f"{m.get('f1_distractor', 0):.4f}")
            note = snap.get("ranking_at_3_note")
            if note:
                st.info(note)
            with st.expander("Raw JSON"):
                st.json(snap)

    with inner[2]:
        st.write(f"Entries: **{len(st.session_state.session_log)}**")
        logs = st.session_state.session_log[::-1]
        if logs:
            st.dataframe(logs[:40], use_container_width=True, hide_index=True)
        export_logs_csv()


def render_model_pills(flags: Optional[Dict[str, Any]]):
    if not flags:
        return
    parts = []
    labels = [
        ("model_a", "Model A LR"),
        ("svm_calibrated", "SVM (ensemble)"),
        ("qg_ranker", "QG ranker"),
        ("distractor_ranker", "Distractors"),
        ("hint_ranker", "Hints"),
        ("phase4_results", "Phase 4 JSON"),
        ("stacking_meta_lr", "Stacking pkl"),
    ]
    for key, label in labels:
        ok = flags.get(key, False)
        cls = "rc-pill rc-on" if ok else "rc-pill rc-off"
        parts.append(f'<span class="{cls}">{label}: {"on" if ok else "off"}</span>')
    st.markdown("<div>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def sidebar():
    st.sidebar.markdown("### Quiz settings")
    st.session_state.distractor_lambda = st.sidebar.slider(
        "Distractor diversity (λ)",
        min_value=0.2,
        max_value=0.8,
        value=float(st.session_state.distractor_lambda),
        step=0.05,
        help="Higher λ pushes the three wrong options to be more different from each other (MMR).",
    )
    st.session_state.force_generate = st.sidebar.checkbox(
        "Always build MCQ from passage",
        value=st.session_state.force_generate,
        help="When off, a loaded RACE row reuses its question and four options; hints still use your hint model.",
    )
    st.session_state.use_question_generator = st.sidebar.checkbox(
        "Use Phase 3 ML question generator (qg_ranker)",
        value=st.session_state.get("use_question_generator", True),
        help="WH-templates + RandomForest ranker when qg_ranker.pkl is present; otherwise rotate template stems.",
    )
    idx_ver = 0 if st.session_state.get("verification_ensemble", "soft_ls") == "soft_ls" else 1
    choice = st.sidebar.selectbox(
        "Answer verification",
        options=["soft_ls", "lr"],
        index=idx_ver,
        format_func=lambda x: "Soft ensemble (LR + SVM mean)" if x == "soft_ls" else "Logistic regression only",
        help="OHE + lexical + cosine (Phase 2). Requires svm_calibrated.pkl for soft ensemble.",
    )
    st.session_state.verification_ensemble = choice

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Models on disk")
    disk = inference.models_available()
    for name, ok in disk["loaded"].items():
        st.sidebar.write(f"{'✓' if ok else '✗'} `{name}`")
    with st.sidebar.expander("Artifact paths"):
        for k, v in disk["paths"].items():
            st.code(v, language=None)

    st.sidebar.markdown("---")
    st.sidebar.caption("Run from project root: `streamlit run ui/app.py`")


def screen_input():
    st.subheader("1 · Passage")
    st.caption("Paste text or load a random RACE row. Submit builds the quiz with your saved models.")
    st.session_state.article_text = st.text_area(
        "Article",
        value=st.session_state.article_text,
        height=300,
        label_visibility="collapsed",
        placeholder="Paste a reading passage here…",
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Load random RACE sample", use_container_width=True):
            sample = load_race_sample()
            if sample:
                st.session_state.article_text = sample.get("article", "")
                st.session_state._race_row = sample
                st.session_state.force_generate = False
                st.success("Loaded test split row — submit to use its MCQ (or enable “build from passage”).")
            else:
                st.warning("Missing `data/raw/test.csv`")
    with c2:
        if st.button("Clear RACE pairing", use_container_width=True):
            st.session_state.pop("_race_row", None)
            st.info("MCQ will be generated from the passage only.")
    with c3:
        run = st.button("Generate quiz", type="primary", use_container_width=True)

    if not run:
        return

    if not st.session_state.get("article_text", "").strip():
        st.error("Add a passage first.")
        return

    race_row = None
    if st.session_state.get("_race_row") and not st.session_state.force_generate:
        race_row = st.session_state._race_row

    with st.spinner("Running Model A + distractor + hint pipelines…"):
        t0 = time.time()
        st.session_state.quiz_stem_generation = int(st.session_state.quiz_stem_generation) + 1
        result = inference.run_pipeline(
            st.session_state.article_text,
            race_row=race_row,
            distractor_lambda=st.session_state.distractor_lambda,
            quiz_stem_rotate_key=st.session_state.quiz_stem_generation,
            use_question_generator=st.session_state.get("use_question_generator", True),
            verification_ensemble=st.session_state.get("verification_ensemble", "soft_ls"),
        )
        latency = (time.time() - t0) * 1000
    result["latency_ms"] = latency
    st.session_state.last_result = result
    st.session_state.hints_opened = 0
    probs = result.get("option_probs")
    probs_json = None
    if probs is not None:
        try:
            probs_json = json.dumps([float(x) for x in probs])
        except (TypeError, ValueError):
            probs_json = None
    ci = result.get("correct_index")
    pci = result.get("predicted_correct_index")
    add_log(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "article_length_words": len(st.session_state.article_text.split()),
            "question": result.get("question"),
            "quiz_mode": result.get("quiz_mode"),
            "latency_ms": round(latency, 1),
            "correct_index": ci,
            "predicted_correct_index": pci,
            "verification_backend": result.get("verification_backend"),
            "model_a_match": pci == ci if pci is not None and ci is not None else None,
            "option_probs_json": probs_json,
            "user_choice_index": None,
            "was_correct": None,
        }
    )
    st.success(f"Done in {latency:.0f} ms — open **Quiz**.")
    render_model_pills(result.get("models"))


def screen_quiz():
    res = st.session_state.last_result
    if not res:
        st.info("Generate a quiz from **Article** first.")
        return

    st.subheader("2 · Question")
    mode = res.get("quiz_mode", "")
    qsrc = res.get("question_source", "")
    st.caption(
        f"Mode: **{mode}** · Question source: **{qsrc or '—'}** · "
        f"Verifier: **`{res.get('verification_backend', '—')}`** · "
        f"Latency **{res.get('latency_ms', 0):.0f} ms**"
    )
    st.markdown(res["question"])
    if res.get("question_anchor"):
        with st.expander("Anchor sentence (question generation)"):
            st.write(res["question_anchor"])
    if res.get("stacking_note"):
        st.info(res["stacking_note"])
    p4 = res.get("phase4_summary")
    if p4:
        with st.expander("Phase 4 (unsupervised / semi-supervised) saved results"):
            st.json(p4.get("summary", p4))

    options = list(res["options"])
    labels = [f"**{chr(65 + i)}.** {opt}" for i, opt in enumerate(options)]
    choice_idx = st.radio(
        "Select an option",
        range(4),
        format_func=lambda i: labels[i],
        horizontal=False,
    )

    if st.button("Check answer", type="primary"):
        correct_idx = res.get("correct_index", 0)
        was_correct = choice_idx == correct_idx
        if st.session_state.session_log:
            st.session_state.session_log[-1]["user_choice_index"] = int(choice_idx)
            st.session_state.session_log[-1]["was_correct"] = was_correct
        if was_correct:
            st.success("Correct — nice work.")
        else:
            st.error("Not quite — try a hint or compare with the passage.")
        if res.get("explanation"):
            with st.expander("Model A · verification scores", expanded=False):
                st.markdown(res["explanation"])
        probs = res.get("option_probs")
        if probs and len(probs) == 4:
            st.bar_chart(
                pd.DataFrame({"Verification score": probs}, index=["A", "B", "C", "D"])
            )

    st.subheader("Passage (reference)")
    st.text_area("copy", st.session_state.article_text, height=160, disabled=True, label_visibility="collapsed")


def screen_hints():
    res = st.session_state.last_result
    if not res:
        st.info("Generate a quiz first.")
        return
    st.subheader("3 · Hints")
    hints = res.get("hints", ["", "", ""])
    for i, hint in enumerate(hints, start=1):
        with st.expander(f"Hint {i} (gentler → stronger)", expanded=False):
            st.write(hint)
            if st.button(f"Mark hint {i} read", key=f"hintread_{i}"):
                st.session_state.hints_opened = max(st.session_state.hints_opened, i)
    if st.session_state.hints_opened >= 3:
        ci = res.get("correct_index", 0)
        st.info(f"Answer: **{res['options'][ci]}**")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    init_session()
    st.title(APP_TITLE)
    st.caption("Reading comprehension quiz — integrated with your trained traditional models.")
    sidebar()

    tab_labels = ["Article", "Quiz", "Hints", "Analytics"]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        screen_input()
    with tabs[1]:
        screen_quiz()
    with tabs[2]:
        screen_hints()
    with tabs[3]:
        screen_analytics()


if __name__ == "__main__":
    main()
