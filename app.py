from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
import streamlit as st

# ── 最初の Streamlit コマンド ──
st.set_page_config(page_title="LLMアプリ（教材提出）", layout="centered")

# ── APIキー読み込み：.env（ローカル）→ secrets（Cloud） ──
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY が見つかりません。.env（ローカル）または Streamlit Secrets（Cloud）に設定してください。")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── LLM ──
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=60)

# ── ロール定義（2択）──
EXPERT_ROLES = {
    "品質保証エンジニア（製造業）": (
        "あなたは製造業の品質保証（QA）と統計的品質管理の専門家です。"
        "工程能力(Cp/Cpk)、不良モード、FMEA、QC七つ道具、PDCAに精通し、"
        "入力内容に対して根拠・手順・数式・注意点・現実的制約を併記しつつ、"
        "再現可能な改善策とチェックリストを提示してください。"
    ),
    "営業のプロ（B2B/B2C）": (
        "あなたはB2B/B2Cに精通した営業のプロです。"
        "ヒアリング設計、課題深掘り（SPIN）、価値訴求（FABE/ベネフィット）、"
        "差別化、反論処理、次アクション合意、クロージングまでを体系的に提案し、"
        "トークスクリプト例・質問リスト・KPI/指標・テンプレを具体的に提示してください。"
        "顧客セグメントや意思決定者/使用者の切り分けにも触れてください。"
    ),
}

# ── Prompt（履歴をMessagesPlaceholderで注入）──
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input_text}"),
    ]
)
base_chain = prompt | llm

# ── セッション履歴 ──
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
if "history_store" not in st.session_state:
    st.session_state.history_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    store = st.session_state.history_store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ── RunnableWithMessageHistory ──
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input_text",
    history_messages_key="history",
)

# ── 通常呼び出し（非ストリーミング）──


def ask_llm_once(input_text: str, role_key: str) -> str:
    system_message = EXPERT_ROLES.get(role_key, "")
    if not system_message:
        raise ValueError("不明な専門家ロールが選択されました。")
    if not input_text or not input_text.strip():
        raise ValueError("入力テキストが空です。")

    result = chain_with_history.invoke(
        {"system_message": system_message, "input_text": input_text.strip()},
        config={"configurable": {"session_id": st.session_state.session_id}},
    )
    return result.content


# ================= UI =================
st.title(" LLM機能付きWebアプリ")
with st.expander("📘 このアプリについて（概要・操作方法）", expanded=True):
    st.markdown(
        """
**概要**  
- Streamlit + LangChain の LLMアプリ。  
- ラジオ選択の専門家ロールに応じて System メッセージを切替。  
- **会話履歴を踏まえた回答**を **通常出力**でまとめて表示します。  

**操作方法**  
1. ロールを選択  
2. テキストを入力  
3. 「送信」で回答表示  
4. 「履歴クリア」で会話履歴をリセット  
        """
    )

role = st.radio("専門家ロールを選択してください", options=list(
    EXPERT_ROLES.keys()), horizontal=True)
user_text = st.text_area(
    "入力テキスト",
    value="新規ラインの不良率を下げたい。現場で今日からできる打ち手を教えて。（例）",
    height=140,
    placeholder="ここに相談内容または質問を入力してください…",
)

cols = st.columns([1, 1, 2])
with cols[0]:
    submitted = st.button("送信", use_container_width=True)
with cols[1]:
    if st.button("履歴クリア", use_container_width=True):
        st.session_state.history_store.pop(st.session_state.session_id, None)
        st.success("履歴をクリアしました。")
        st.rerun()

# 履歴の可視化
with st.expander("🗂 会話履歴（このセッション）"):
    hist = get_session_history(st.session_state.session_id).messages
    if hist:
        for m in hist:
            who = "👤 User" if m.type == "human" else "AI"
            st.markdown(f"- **{who}**: {m.content}")
    else:
        st.caption("まだ履歴はありません。")

if submitted:
    try:
        with st.spinner("LLMに問い合わせ中…"):
            answer = ask_llm_once(user_text, role)
        st.success("回答を取得しました。")
        st.markdown("### ✅ 回答")
        st.write(answer)
    except Exception as e:
        st.error(f"エラーが発生しました：{e}")
