"""Minimal Streamlit shell for the citation-verification prototype."""

from __future__ import annotations

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.citation_parser import CitationParseError
from src.jev_adapter import OpenRouterJevAdapter
from src.method_b import DEFAULT_METHOD_B_MODEL, run_method_b
from src.mock_adapters import MockJevAdapter, MockReferenceLookupAdapter
from src.mock_data import MOCK_CLAIM, MOCK_REFERENCES
from src.pairing import build_pairing_outcome
from src.pipeline import run_method_a_pipeline
from src.presentation import (
    build_output_dataframe,
    dataframe_to_csv_bytes,
    style_output_dataframe,
    summarize_shared_claims,
)
from src.reference_lookup import SCHOLAR_REQUEST_SCHEDULE, ReferenceLookupAdapter, RequestsClient
from src.schemas import AnalysisRow, CitationPair, ReferenceMatchStatus
from src.validation import IssueSeverity, validate_required_inputs


st.set_page_config(
    page_title="未獲來源支持引用偵測系統",
    page_icon="🔎",
    layout="wide",
)


def initialize_state() -> None:
    st.session_state.setdefault("claim_text", MOCK_CLAIM)
    st.session_state.setdefault("reference_text", MOCK_REFERENCES)
    st.session_state.setdefault("show_results", False)
    st.session_state.setdefault("analysis_rows", [])
    st.session_state.setdefault("result_issues", [])
    st.session_state.setdefault("result_mode", "")
    st.session_state.setdefault("reference_cooldowns", {})
    st.session_state.setdefault("scholar_abstracts", {})
    st.session_state.setdefault("use_mock_pipeline", True)


def clear_form() -> None:
    st.session_state.claim_text = ""
    st.session_state.reference_text = ""
    st.session_state.show_results = False
    st.session_state.analysis_rows = []
    st.session_state.result_issues = []
    st.session_state.result_mode = ""


def read_api_key(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except StreamlitSecretNotFoundError:
        return ""
    return value.strip() if isinstance(value, str) else ""


def analyze_pairs(pairs: list[CitationPair]) -> tuple[list[AnalysisRow], str]:
    if st.session_state.use_mock_pipeline:
        rows = run_method_a_pipeline(pairs, MockReferenceLookupAdapter(), MockJevAdapter())
        return rows, "mock"

    api_key = read_api_key("OPENROUTER_API_KEY")
    if not api_key:
        return [AnalysisRow(pair=pair) for pair in pairs], "missing_key"

    reference_adapter = ReferenceLookupAdapter(
        s2_key=read_api_key("S2_API_KEY"),
        openalex_key=read_api_key("OPENALEX_API_KEY"),
        http=RequestsClient(
            cooldowns=st.session_state.reference_cooldowns,
            scholar_schedule=SCHOLAR_REQUEST_SCHEDULE,
            request_source=os.environ.get("CITATION_REQUEST_SOURCE", "streamlit-app"),
        ),
        scholar_cache=st.session_state.scholar_abstracts,
    )
    rows = run_method_a_pipeline(
        pairs, reference_adapter, OpenRouterJevAdapter(
            api_key, include_evidence_excerpt=True, resolve_abstract_title=True,
        ),
    )
    rows = run_method_b(
        rows, st.session_state.claim_text, st.session_state.reference_text, api_key,
        model=os.environ.get("OPENROUTER_METHOD_B_MODEL") or DEFAULT_METHOD_B_MODEL,
        evidence_lookup=reference_adapter.lookup,
    )
    return rows, "jev"


initialize_state()

st.title("未獲來源支持引用偵測系統")
st.caption("第一階段原型：人工輸入、限定 IEEE 格式、全文暫不處理")

st.progress(0.20, text="輸入 → 擷取引用 → 比對 → 結果 → 下載")

with st.container(border=True):
    st.subheader("輸入資料")
    use_mock_pipeline = st.toggle(
        "開發測試模式：使用 mock 摘要與 Jev 結果",
        key="use_mock_pipeline",
        help="此模式不會呼叫外部 API；結果只用來測試串接與畫面。",
    )
    if not use_mock_pipeline:
        st.caption("正式比對時，方法 A 會將引用句和證據送到 OpenRouter，方法 B 會送出貼上的段落與書目。")
    st.text_area(
        "① 欲比對的段落",
        key="claim_text",
        height=140,
        help="請貼上含有 IEEE 文內引用編號（例如 [79]）的段落。",
    )
    st.text_area(
        "② 參考文獻清單（ref）",
        key="reference_text",
        height=260,
        help="請貼上論文最後的參考文獻清單，不用先挑選。",
    )

    left, right = st.columns([1, 1])
    with left:
        st.button("清除", on_click=clear_form, width="stretch")
    with right:
        if st.button("開始比對", type="primary", width="stretch"):
            input_issues = validate_required_inputs(
                st.session_state.claim_text,
                st.session_state.reference_text,
            )
            if input_issues:
                for issue in input_issues:
                    st.error(issue.message)
                st.session_state.show_results = False
            else:
                try:
                    outcome = build_pairing_outcome(
                        st.session_state.claim_text,
                        st.session_state.reference_text,
                    )
                except CitationParseError as error:
                    st.error(str(error))
                    st.session_state.show_results = False
                else:
                    with st.spinner("正在比對引用，請稍候。"):
                        rows, result_mode = analyze_pairs(outcome.pairs)
                    st.session_state.analysis_rows = rows
                    st.session_state.result_mode = result_mode
                    st.session_state.result_issues = outcome.issues
                    st.session_state.show_results = True

if st.session_state.show_results:
    st.divider()
    st.subheader("引用配對結果")
    rows: list[AnalysisRow] = st.session_state.analysis_rows
    result_issues = st.session_state.result_issues
    for issue in result_issues:
        if issue.severity is IssueSeverity.ERROR:
            st.error(issue.message)
        else:
            st.warning(issue.message)
    if not rows:
        if not result_issues:
            st.warning("段落中沒有偵測到可解析的 IEEE 數字引用。")
    else:
        shared_claims = summarize_shared_claims(rows)
        missing_count = sum(
            row.pair.match_status is ReferenceMatchStatus.MISSING_REFERENCE
            for row in rows
        )
        st.write(f"偵測到 {len(rows)} 個引用配對")
        if missing_count:
            st.warning(f"其中 {missing_count} 筆在 ref 清單中找不到對應條目。")
        result_mode = st.session_state.result_mode
        if result_mode == "mock":
            st.warning(
                "目前方法 A 顯示的是 MOCK 摘要與 Jev 結果，只供串接測試；"
                "方法 B 在開發測試模式保持空白。"
            )
        elif result_mode == "missing_key":
            st.warning("尚未設定 OPENROUTER_API_KEY，目前只顯示引用配對。")
        elif result_mode == "jev":
            st.info("方法 A 使用 Jev，方法 B 使用一般 LLM 直接判讀貼上的書目。")
        if shared_claims:
            st.subheader("方法 A 共引子主張涵蓋")
            for group in shared_claims:
                st.text(group.original_claim)
                st.write(
                    f"摘要支持 {group.supported}/{group.total} 項子主張"
                    f" · 待查 {group.pending} 項"
                )
            st.caption("下表每列列出一篇候選文獻對一項子主張的判斷。涵蓋數以摘要支持為準。")
        st.markdown(
            "**證據層級：** "
            "<span style='background:#E8F5E9;color:#1B5E20;padding:3px 10px;"
            "border-radius:12px;font-weight:600'>支持</span> "
            "<span style='background:#FFF3E0;color:#A65300;padding:3px 10px;"
            "border-radius:12px;font-weight:600'>不確定</span> "
            "<span style='background:#FDECEC;color:#B42318;padding:3px 10px;"
            "border-radius:12px;font-weight:600'>不支持</span> "
            "<span style='background:#F2F4F7;color:#475467;padding:3px 10px;"
            "border-radius:12px;font-weight:600'>無法查核</span>",
            unsafe_allow_html=True,
        )
        frame = build_output_dataframe(rows)
        has_subclaims = frame["查核子主張"].ne("").any()
        table_columns = frame.columns.tolist()
        if has_subclaims:
            table_columns.remove("查核子主張")
            table_columns.insert(2, "查核子主張")
        comparison_columns = [
            "方法 A 證據解讀", "方法 A 摘要節錄",
            "方法 B 支持分數", "方法 B 證據層級", "方法 B 判斷依據", "方法 B 判斷狀態",
        ]
        for column in comparison_columns:
            table_columns.remove(column)
        comparison_start = table_columns.index("方法 A 證據層級") + 1
        table_columns[comparison_start:comparison_start] = comparison_columns
        st.dataframe(
            style_output_dataframe(frame),
            hide_index=True,
            width="stretch",
            column_order=table_columns,
            column_config={
                "引用陳述": st.column_config.TextColumn(width="large"),
                "查核子主張": st.column_config.TextColumn(width="medium") if has_subclaims else None,
                "共引群組": None,
                "共引摘要涵蓋": None,
                "共引待查數": None,
                "被引文獻": st.column_config.TextColumn(width="large"),
                "摘要查詢紀錄": st.column_config.TextColumn(width="medium"),
                "摘要來源網址": st.column_config.LinkColumn(),
                "摘要 HTTP 請求紀錄": None,
                "方法 A 支持分數": st.column_config.NumberColumn(format="plain"),
                "方法 B 支持分數": st.column_config.NumberColumn(format="plain"),
                "方法 A 查核紀錄": None,
                "方法 A 證據解讀": st.column_config.TextColumn(width="medium"),
                "方法 A 摘要節錄": st.column_config.TextColumn(
                    width="large", help="Jev 從原摘要選出的相關節錄，可用來核對原句的一部分。整句分級仍由 Q2 決定。",
                ),
                "方法 B 判斷狀態": st.column_config.TextColumn(width="medium"),
                "方法 B 查核紀錄": None,
            },
        )
        st.caption(
            "支持 ≥ 0.70 · 不確定 ≥ 0.40 且 < 0.70 · 不支持 < 0.40。"
            "方法 A 篇名不相關時直接列為不支持。拿不到摘要時兩個方法都列為無法查核、不給分數。"
            "兩個方法用同一份摘要、同一套判斷標準，只判斷句子歸給該文獻的部分。"
        )
        st.download_button(
            "匯出 CSV",
            data=dataframe_to_csv_bytes(frame),
            file_name="citation_pairing_results.csv",
            mime="text/csv",
            type="primary",
            help="下載 A、B 的逐筆結果與查核紀錄。",
        )



