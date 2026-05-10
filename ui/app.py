import streamlit as st
import time
import csv
import io
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Ensure project root is on sys.path so sibling packages (src) can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import inference


APP_TITLE = "Reading-Comprehension — Quiz UI"


def load_race_sample(csv_path: str = 'data/raw/test.csv'):
    p = Path(csv_path)
    if not p.exists():
        return None
    import random
    with p.open('r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return None
        return random.choice(reader)


def init_session():
    if 'session_log' not in st.session_state:
        st.session_state.session_log = []
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'hints_opened' not in st.session_state:
        st.session_state.hints_opened = 0
    if 'article_text' not in st.session_state:
        st.session_state.article_text = ''


def add_log(entry: Dict[str, Any]):
    st.session_state.session_log.append(entry)


def export_logs_csv() -> None:
    if not st.session_state.session_log:
        st.info('No logs yet')
        return
    output = io.StringIO()
    keys = list(st.session_state.session_log[0].keys())
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    for row in st.session_state.session_log:
        writer.writerow(row)
    st.download_button('Download session log CSV', output.getvalue(), file_name='session_log.csv')


def screen_input():
    st.header('Article Input')
    st.write('Paste an article or load a random RACE sample for testing.')
    st.session_state.article_text = st.text_area('Article', value=st.session_state.article_text, height=280)
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button('Load random RACE sample'):
            sample = load_race_sample()
            if sample:
                st.session_state.article_text = sample.get('article', '')
                st.session_state._gold_q = sample
                st.success('Loaded a random sample (test split)')
            else:
                st.warning('No test.csv found at data/raw/test.csv')
    with col2:
        if st.button('Submit'):
            if not st.session_state.get('article_text'):
                st.error('Please paste an article before submitting.')
                return
            with st.spinner('Running inference (Model A & B)...'):
                t0 = time.time()
                result = inference.run_pipeline(st.session_state.article_text)
                latency = (time.time() - t0) * 1000
                result['latency_ms'] = latency
                st.session_state.last_result = result
                st.session_state.hints_opened = 0
                add_log({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'article_length_words': len(st.session_state.article_text.split()),
                    'question': result.get('question'),
                    'latency_ms': round(latency, 1),
                    'was_correct': None
                })
                st.success('Inference complete — go to Quiz tab')


def screen_quiz():
    res = st.session_state.last_result
    if not res:
        st.info('Run the Article Input screen first to generate a question.')
        return
    st.header('Question & Quiz')
    st.markdown(f"**Question:** {res['question']}")
    options = list(res['options'])
    choice = st.radio('Choose an option', options)
    if st.button('Check Answer'):
        correct_idx = res.get('correct_index', 0)
        chosen_idx = options.index(choice)
        was_correct = (chosen_idx == correct_idx)
        if st.session_state.session_log:
            st.session_state.session_log[-1]['was_correct'] = was_correct
        if was_correct:
            st.success('✅ Correct!')
        else:
            st.error('❌ Incorrect. Try hints or review the passage.')
        explanation = res.get('explanation')
        if explanation:
            st.info(explanation)


def screen_hints():
    res = st.session_state.last_result
    if not res:
        st.info('Run the Article Input screen first to generate a question.')
        return
    st.header('Hints')
    hints = res.get('hints', ['','',''])
    for i, hint in enumerate(hints, start=1):
        with st.expander(f'Hint {i}', expanded=False):
            st.write(hint)
            if st.button(f'Mark Hint {i} as read'):
                st.session_state.hints_opened = max(st.session_state.hints_opened, i)
    if st.session_state.hints_opened >= 3:
        if st.button('Reveal Answer'):
            st.info(f"Correct answer: {res['options'][res.get('correct_index',0)]}")


def screen_dashboard():
    st.header('Developer / Analytics Dashboard')
    st.subheader('Recent Inferences')
    logs = st.session_state.session_log[::-1]
    st.write(f'Total logged inferences: {len(st.session_state.session_log)}')
    if logs:
        st.table(logs[:10])
    st.subheader('Export')
    export_logs_csv()


def main():
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    init_session()
    st.title(APP_TITLE)
    st.sidebar.markdown('## Navigation')
    menu = st.sidebar.radio('', ['Article Input', 'Quiz', 'Hints', 'Dashboard'])
    if menu == 'Article Input':
        screen_input()
    elif menu == 'Quiz':
        screen_quiz()
    elif menu == 'Hints':
        screen_hints()
    else:
        screen_dashboard()


if __name__ == '__main__':
    main()
