from __future__ import annotations

import csv
from http.client import IncompleteRead
from io import BytesIO, StringIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from streamlit.testing.v1 import AppTest

from src.adapters import JudgmentStatus, ReferenceLookupResult
from src.jev_adapter import DECISIONS_URL, DEFAULT_MODEL, OpenRouterJevAdapter
from src.mock_data import MOCK_CLAIM, MOCK_REFERENCES
from src.pairing import build_citation_pairs
from src.pipeline import run_method_a_pipeline
from src.presentation import build_output_dataframe, dataframe_to_csv_bytes, summarize_shared_claims
from src.reference_lookup import OPENALEX_BASE, S2_BASE
from src.schemas import CitationPair, ReferenceMatchStatus, SourceStatus, SupportBand


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAIR = CitationPair(
    "pair_001",
    "The method reduces treatment time [1].",
    "[1]",
    1,
    '[1] A. Author, "A study of treatment time," 2026.',
)
TITLE = ReferenceLookupResult(1, "A study of treatment time")


def title_answer(choice: str) -> dict:
    probabilities = dict.fromkeys(("related", "uncertain", "unrelated"), 0.1)
    probabilities[choice] = 0.8
    return {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.6}


class DecisionsStub:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, *answers: dict | Exception):
        self.answers = iter(answers)
        self.states = []
        self.events = []
        monkeypatch.setattr("src.jev_adapter.urlopen", self.open)

    def open(self, request, *, timeout):
        payload = json.loads(request.data)
        assert request.full_url == DECISIONS_URL
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer test-placeholder"
        assert request.get_header("Content-type") == "application/json"
        assert payload["model"] == DEFAULT_MODEL
        assert timeout > 0
        [(question_id, question)] = payload["questions"].items()
        self.events.append(question_id)
        self.states.append(payload["state"])
        if question_id == "title_relevance":
            assert set(question["criteria"]) == {"related", "uncertain", "unrelated"}
        elif question_id == "evidence_excerpt":
            assert question["type"] == "choice"
            assert set(question["criteria"]) == set(payload["state"]) - {"claim"} | {"none"}
        else:
            assert question["type"] == "noul"
            assert "`claim`" in question["instructions"]
            assert "`evidence`" in question["instructions"]

        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        response = {
            "model": DEFAULT_MODEL,
            "id": f"call-{len(self.states)}",
            "answers": {question_id: answer},
        }
        return BytesIO(json.dumps(response).encode("utf-8"))


@pytest.mark.parametrize(
    ("answer", "excerpt", "status"),
    [
        ({"type": "choice", "choice": "excerpt_2"}, "Measurements show the operation took seconds.",
         "摘要有可核對的相關內容"),
        ({"type": "choice", "choice": "none"}, "", "未辨識到可核對的節錄"),
        ({"type": "choice", "choice": "invented_text"}, "", "節錄檢查未完成"),
        (TimeoutError(), "", "節錄檢查未完成"),
    ],
)
def test_aside_from_q2_only_verbatim_abstract_excerpts_can_be_selected(monkeypatch, answer, excerpt, status):
    stub = DecisionsStub(monkeypatch, title_answer("related"), {"type": "noul", "noul": 0.08}, answer)
    ref = ReferenceLookupResult(
        1, TITLE.title, abstract="The researchers benchmarked a method. Measurements show the operation took seconds.",
        source_status=SourceStatus.GOT_ABSTRACT,
    )
    adapter = OpenRouterJevAdapter("test-placeholder", include_evidence_excerpt=True)

    judgment = adapter.judge(PAIR, ref)

    assert stub.events == ["title_relevance", "citation_support", "evidence_excerpt"]
    assert stub.states[1]["claim"] == PAIR.claim_text
    assert stub.states[1]["evidence"] == ref.abstract
    assert judgment.raw_support_score == 0.08
    assert judgment.support_score == 0.10
    assert judgment.support_request_id == "call-2"
    assert judgment.evidence_excerpt == excerpt
    assert judgment.evidence_excerpt_status == status


@pytest.mark.parametrize(
    ("papers", "expected_supported", "expected_pending"),
    [
        (("BERT", "SciBERT"), 2, 0),
        (("SciBERT", "BERT"), 2, 0),
        (("BERT", "Unrelated"), 1, 0),
        (("BERT", "Lookup failure"), 1, 1),
        (("BERT", "Title only"), 1, 1),
    ],
)
def test_shared_claim_coverage_uses_evidence_instead_of_reference_order(
    monkeypatch, papers, expected_supported, expected_pending,
):
    # 這個測試驗的是共引句拆子主張的行為，預設關閉，這裡明確開啟
    monkeypatch.setattr("src.pairing.SPLIT_SHARED_SUBCLAIMS", True)
    source_sentence = "BERT and SciBERT are pretrained language models [1], [2]."
    refs = "\n".join(f'[{index}] A. Author, "{paper}," 2019.' for index, paper in enumerate(papers, 1))
    subclaims = ("BERT is a pretrained language model.", "SciBERT is a pretrained language model.")
    answers = []
    for paper in papers:
        for model in ("BERT", "SciBERT"):
            answers.append(title_answer("unrelated" if paper == "Unrelated" else "related"))
            if paper != "Unrelated":
                score = 0.98 if paper == model or paper in ("Title only", "Lookup failure") else 0.1
                answers.append({"type": "noul", "noul": score})
    stub = DecisionsStub(monkeypatch, *answers)

    def lookup(pair):
        paper = papers[pair.ref_number - 1]
        if paper == "Lookup failure":
            return ReferenceLookupResult(pair.ref_number, paper, source_status=SourceStatus.LOOKUP_FAILED)
        if paper == "Title only":
            return ReferenceLookupResult(pair.ref_number, paper, source_status=SourceStatus.NO_ABSTRACT)
        return ReferenceLookupResult(pair.ref_number, paper, abstract=f"Abstract for {paper}.",
                                     source_status=SourceStatus.GOT_ABSTRACT)

    reference_adapter = Mock()
    reference_adapter.lookup.side_effect = lookup
    pairs = build_citation_pairs(source_sentence, refs)
    rows = run_method_a_pipeline(pairs, reference_adapter, OpenRouterJevAdapter("test-placeholder"))

    assert len(rows) == 4
    assert [state["claim"] for state in stub.states if "title" in state] == list(subclaims) * 2
    assert [row.reference.title for row in rows if row.method_a.band is SupportBand.SUPPORTED
            and row.reference.abstract] == [paper for paper in papers if paper in ("BERT", "SciBERT")]
    [coverage] = summarize_shared_claims(rows)
    assert (coverage.total, coverage.supported, coverage.pending) == (2, expected_supported, expected_pending)
    if papers[1] == "Unrelated":
        assert [call.args[0].ref_number for call in reference_adapter.lookup.call_args_list] == [1, 1]
    if papers[1] == "Lookup failure":
        assert all(row.judgment.status is JudgmentStatus.TITLE_AFTER_LOOKUP_FAILURE for row in rows[2:])
        assert all(row.method_a.band is SupportBand.SUPPORTED for row in rows[2:])

    csv_data = dataframe_to_csv_bytes(build_output_dataframe(rows))
    exported = list(csv.DictReader(StringIO(csv_data.decode("utf-8-sig"))))
    assert {record["引用陳述"] for record in exported} == {source_sentence}
    assert {record["共引群組"] for record in exported} == {coverage.group_id}
    assert [record["查核子主張"] for record in exported] == list(subclaims) * 2
    assert {record["共引摘要涵蓋"] for record in exported} == {f"{expected_supported}/2 項"}
    assert {record["共引待查數"] for record in exported} == {str(expected_pending)}


def test_unrelated_title_skips_lookup_and_q2_and_exports_blank_score(monkeypatch):
    stub = DecisionsStub(monkeypatch, title_answer("unrelated"))
    reference_adapter = Mock()
    rows = run_method_a_pipeline(
        [PAIR], reference_adapter, OpenRouterJevAdapter("test-placeholder"),
    )
    reference_adapter.lookup.assert_not_called()
    assert stub.events == ["title_relevance"]
    assert stub.states[0] == {"title": TITLE.title, "claim": PAIR.claim_text}
    row = rows[0]
    assert row.judgment.title_probabilities == {"相關": 0.1, "不確定": 0.1, "不相關": 0.8}
    assert row.judgment.confidence == 0.6
    csv_bytes = dataframe_to_csv_bytes(build_output_dataframe(rows))
    [record] = csv.DictReader(StringIO(csv_bytes.decode("utf-8-sig")))
    assert record["方法 A 支持分數"] == ""
    assert record["方法 A 證據層級"] == "不支持"
    assert record["方法 A 判斷依據"] == "篇名"
    assert record["方法 A 判斷狀態"] == "Q1 停止"
    assert record["來源狀態"] == "Q1 停止未查詢"
    assert record["方法 B 證據層級"] == ""


@pytest.mark.parametrize(
    ("choice", "abstract", "source_status", "shown_status", "evidence"),
    [
        ("related", "The study found shorter treatment times.", SourceStatus.GOT_ABSTRACT,
         "拿到摘要", "摘要"),
        ("uncertain", "", SourceStatus.NO_ABSTRACT, "有文獻但沒摘要", "篇名"),
        ("uncertain", "", SourceStatus.NOT_FOUND, "查不到", "篇名"),
        ("uncertain", "", SourceStatus.LOOKUP_FAILED, "查詢失敗", "篇名"),
    ],
)
def test_abstract_lookup_runs_between_q1_and_q2(
    monkeypatch, choice, abstract, source_status, shown_status, evidence
):
    stub = DecisionsStub(monkeypatch, title_answer(choice), {"type": "noul", "noul": 0.699})
    reference_adapter = Mock()

    def lookup(pair):
        stub.events.append("lookup")
        assert pair == PAIR
        return ReferenceLookupResult(
            1, TITLE.title, doi="10.5555/test", abstract=abstract, source_status=source_status,
            abstract_source="semantic_scholar" if abstract else "",
            source_note="資料庫摘要查詢紀錄",
            error_message="查詢未完成" if source_status is SourceStatus.LOOKUP_FAILED else "",
        )

    reference_adapter.lookup.side_effect = lookup
    [row] = run_method_a_pipeline(
        [PAIR], reference_adapter, OpenRouterJevAdapter("test-placeholder"),
        prepare_reference=lambda pair: TITLE,
    )
    assert stub.events == ["title_relevance", "lookup", "citation_support"]
    assert stub.states[1] == {
        "claim": PAIR.claim_text, "evidence_type": evidence, "evidence": abstract or TITLE.title,
    }
    assert row.judgment.raw_support_score == 0.699
    assert row.judgment.status is (
        JudgmentStatus.TITLE_AFTER_LOOKUP_FAILURE if source_status is SourceStatus.LOOKUP_FAILED
        else JudgmentStatus.COMPLETED
    )
    assert row.judgment.title_request_id == "call-1"
    assert row.judgment.support_request_id == "call-2"
    assert row.reference.doi == "10.5555/test"
    assert row.reference.abstract == abstract
    record = row.to_record()
    assert record["來源狀態"] == shown_status
    assert record["方法 A 支持分數"] == 0.699
    assert record["方法 A 證據層級"] == "不確定"
    assert record["方法 A 判斷依據"] == evidence
    assert record["摘要查詢紀錄"] == "資料庫摘要查詢紀錄"
    csv_data = dataframe_to_csv_bytes(build_output_dataframe([row]))
    [exported] = csv.DictReader(StringIO(csv_data.decode("utf-8-sig")))
    assert json.loads(exported["方法 A 查核紀錄"]) == {
        "abstract_source": "semantic_scholar" if abstract else "",
        "abstract_url": "",
        "evidence": stub.states[1]["evidence"],
        "model": DEFAULT_MODEL,
        "request_id": "call-2",
        "raw_score": 0.699,
    }


@pytest.mark.parametrize(
    ("raw", "score", "band"),
    [
        (0.0, 0.10, SupportBand.UNSUPPORTED),
        (0.399, 0.399, SupportBand.UNSUPPORTED),
        (0.40, 0.40, SupportBand.UNCERTAIN),
        (0.70, 0.70, SupportBand.SUPPORTED),
        (1.0, 0.99, SupportBand.SUPPORTED),
    ],
)
def test_clip_and_classify_q2_without_rounding(monkeypatch, raw, score, band):
    DecisionsStub(monkeypatch, title_answer("related"), {"type": "noul", "noul": raw})
    [row] = run_method_a_pipeline([PAIR], None, OpenRouterJevAdapter("test-placeholder"))
    assert row.method_a.support_score == score
    assert row.method_a.band is band


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, True, "0.7", None])
def test_invalid_q2_preserves_title_result_and_produces_no_grade(monkeypatch, bad_score):
    DecisionsStub(monkeypatch, title_answer("related"), {"type": "noul", "noul": bad_score})
    [row] = run_method_a_pipeline([PAIR], None, OpenRouterJevAdapter("test-placeholder"))
    assert row.judgment.status is JudgmentStatus.SUPPORT_ERROR
    assert row.judgment.confidence == 0.6
    assert row.method_a.support_score is None
    assert row.method_a.band is None


@pytest.mark.parametrize("failure", [TimeoutError(), IncompleteRead(b"partial")])
def test_failed_q1_preserves_row_and_continues_batch(monkeypatch, failure):
    stub = DecisionsStub(
        monkeypatch, failure, title_answer("related"), {"type": "noul", "noul": 0.71},
    )
    broken, passed = run_method_a_pipeline(
        [PAIR, PAIR], None, OpenRouterJevAdapter("test-placeholder"),
    )
    assert stub.events == ["title_relevance", "title_relevance", "citation_support"]
    assert broken.judgment.status is JudgmentStatus.TITLE_ERROR
    assert broken.method_a.band is None
    assert passed.method_a.band is SupportBand.SUPPORTED


@pytest.mark.parametrize("status", [401, 402])
def test_rejected_key_or_credits_does_not_call_again_in_same_batch(monkeypatch, status):
    failure = HTTPError(DECISIONS_URL, status, "Access failed", None, BytesIO())
    stub = DecisionsStub(monkeypatch, failure, title_answer("unrelated"))
    adapter = OpenRouterJevAdapter("test-placeholder")
    rows = run_method_a_pipeline([PAIR, PAIR], None, adapter)
    assert stub.events == ["title_relevance"]
    assert [row.judgment.status for row in rows] == [
        JudgmentStatus.TITLE_ERROR, JudgmentStatus.BATCH_STOPPED,
    ]
    assert all(row.method_a.band is None for row in rows)
    [retried] = run_method_a_pipeline([PAIR], None, adapter)
    assert stub.events == ["title_relevance", "title_relevance"]
    assert retried.judgment.status is JudgmentStatus.STOPPED


@pytest.mark.parametrize("failed_lookup", [
    KeyError("internal provider details"),
    None,
    ReferenceLookupResult(2, "Another paper", abstract="Wrong abstract"),
    ReferenceLookupResult(1, "Same paper", source_status=SourceStatus.GOT_ABSTRACT),
])
def test_bad_teammate_lookup_uses_only_original_title(monkeypatch, failed_lookup):
    stub = DecisionsStub(monkeypatch, title_answer("related"), {"type": "noul", "noul": 0.55})
    adapter = Mock()
    if isinstance(failed_lookup, Exception):
        adapter.lookup.side_effect = failed_lookup
    else:
        adapter.lookup.return_value = failed_lookup
    [row] = run_method_a_pipeline([PAIR], adapter, OpenRouterJevAdapter("test-placeholder"))
    assert row.method_a.band is SupportBand.UNCERTAIN
    assert row.method_a.source_status is SourceStatus.LOOKUP_FAILED
    assert stub.events == ["title_relevance", "citation_support"]
    assert stub.states[1] == {"claim": PAIR.claim_text, "evidence_type": "篇名", "evidence": TITLE.title}
    assert row.judgment.status is JudgmentStatus.TITLE_AFTER_LOOKUP_FAILURE
    record = row.to_record()
    assert record["方法 A 判斷依據"] == "篇名"
    assert record["方法 A 證據層級"] == "不確定"
    assert "摘要查詢失敗" in record["錯誤狀態"]
    assert "internal provider details" not in record["錯誤狀態"]


def test_bad_metadata_isolated_to_one_pair(monkeypatch):
    stub = DecisionsStub(monkeypatch, title_answer("related"), {"type": "noul", "noul": 0.71})
    prepare = Mock(side_effect=[KeyError("private detail"), TITLE])
    broken, passed = run_method_a_pipeline(
        [PAIR, PAIR], None, OpenRouterJevAdapter("test-placeholder"), prepare_reference=prepare,
    )
    assert stub.events == ["title_relevance", "citation_support"]
    assert broken.judgment.status is JudgmentStatus.REFERENCE_ERROR
    assert "private detail" not in broken.to_record()["錯誤狀態"]
    assert passed.method_a.band is SupportBand.SUPPORTED


def test_meeting_example_runs_with_the_teammate_lookup_interface(monkeypatch):
    stub = DecisionsStub(
        monkeypatch,
        title_answer("unrelated"),
        title_answer("related"), {"type": "noul", "noul": 0.84},
        title_answer("uncertain"), {"type": "noul", "noul": 0.60},
        title_answer("related"), {"type": "noul", "noul": 0.32},
    )

    def lookup(pair):
        stub.events.append(f"lookup:{pair.ref_number}")
        if pair.ref_number == 107:
            return ReferenceLookupResult(
                107, "", abstract="An abstract about credentials.", source_status=SourceStatus.GOT_ABSTRACT,
            )
        return ReferenceLookupResult(pair.ref_number, "", source_status=SourceStatus.NOT_FOUND)

    adapter = Mock()
    adapter.lookup.side_effect = lookup
    pairs = build_citation_pairs(MOCK_CLAIM, MOCK_REFERENCES)
    rows = run_method_a_pipeline(pairs, adapter, OpenRouterJevAdapter("test-placeholder"))
    assert stub.events == [
        "title_relevance",
        "title_relevance", "lookup:107", "citation_support",
        "title_relevance", "lookup:109", "citation_support",
        "title_relevance", "lookup:110", "citation_support",
    ]
    frame = build_output_dataframe(rows)
    assert frame["ref 編號"].tolist() == [47, 107, 109, 110]
    assert frame["方法 A 證據層級"].tolist() == ["不支持", "支持", "不確定", "不支持"]
    assert frame["方法 A 判斷依據"].tolist() == ["篇名", "摘要", "篇名", "篇名"]
    assert (frame["方法 B 證據層級"] == "").all()
    assert rows[1].reference.title == "Verifiable credentials data model v2.0"


def test_rate_limit_after_q1_preserves_title_and_hides_response_body(monkeypatch):
    response_body = BytesIO(b"private upstream error details")
    failure = HTTPError(DECISIONS_URL, 429, "Too Many Requests", None, response_body)
    DecisionsStub(monkeypatch, title_answer("related"), failure)
    [row] = run_method_a_pipeline([PAIR], None, OpenRouterJevAdapter("test-placeholder"))
    assert row.judgment.status is JudgmentStatus.SUPPORT_ERROR
    assert row.judgment.title_choice == "相關"
    assert row.method_a.band is None
    assert "請求受限" in row.judgment.error_message
    assert "private upstream" not in row.judgment.error_message
    assert response_body.closed


def test_missing_title_and_ref_make_no_jev_calls(monkeypatch):
    stub = DecisionsStub(monkeypatch)
    missing_title = CitationPair("pair_001", "Claim [1].", "[1]", 1, "[1] https://example.org")
    missing_ref = CitationPair(
        "pair_002", "Claim [2].", "[2]", 2, "", match_status=ReferenceMatchStatus.MISSING_REFERENCE,
    )
    rows = run_method_a_pipeline(
        [missing_title, missing_ref], None, OpenRouterJevAdapter("test-placeholder"),
    )
    assert stub.events == []
    assert rows[0].judgment.status is JudgmentStatus.MISSING_TITLE
    assert rows[1].method_a is None


def test_missing_key_preserves_pairing_and_uses_no_network(monkeypatch):
    stub = DecisionsStub(monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.secrets["OPENROUTER_API_KEY"] = ""
    app.run()
    app.toggle[0].set_value(False)
    app.button[1].click().run()
    assert not app.exception
    assert "尚未設定" in app.warning[0].value
    assert app.dataframe[0].value.shape[0] == 4
    assert stub.events == []


@pytest.mark.parametrize(
    ("choice", "provider", "source_status", "basis", "band", "sequence"),
    [
        ("related", "semantic_scholar", "拿到摘要", "摘要", "不確定",
         ["title_relevance", "semantic_scholar", "citation_support", "evidence_excerpt"]),
        ("uncertain", "openalex", "拿到摘要", "摘要", "不確定",
         ["title_relevance", "semantic_scholar", "openalex", "citation_support", "evidence_excerpt"]),
        ("uncertain", "", "查不到", "篇名", "不確定",
         ["title_relevance", "semantic_scholar", "openalex", "citation_support"]),
        ("unrelated", "", "Q1 停止未查詢", "篇名", "不支持", ["title_relevance"]),
    ],
)
def test_streamlit_connects_jev_to_real_lookup_adapter_with_stubbed_http(
    monkeypatch, choice, provider, source_status, basis, band, sequence,
):
    stub = DecisionsStub(monkeypatch, title_answer(choice), {"type": "noul", "noul": 0.40},
                        {"type": "choice", "choice": "excerpt_1"})
    scholar_abstract = "Semantic Scholar delivered the abstract."
    openalex_abstract = "OpenAlex delivered the abstract."

    def lookup(url, params=None, headers=None):
        if url == f"{S2_BASE}/paper/search/match":
            assert params["query"] == TITLE.title
            assert headers["x-api-key"] == "scholar-placeholder"
            stub.events.append("semantic_scholar")
            papers = [{"title": TITLE.title, "abstract": scholar_abstract}]
            return 200, {"data": papers if provider == "semantic_scholar" else []}, ""
        assert url == f"{OPENALEX_BASE}/works"
        assert params["search"] == TITLE.title
        assert params["api_key"] == "openalex-placeholder"
        stub.events.append("openalex")
        inverse = {word: [i] for i, word in enumerate(openalex_abstract.split())}
        works = [{"display_name": TITLE.title, "abstract_inverted_index": inverse}]
        return 200, {"results": works if provider == "openalex" else []}, ""

    lookup_client = Mock()
    lookup_client.get.side_effect = lookup
    monkeypatch.setattr("src.reference_lookup.RequestsClient", lambda **kwargs: lookup_client)

    b_requests = []

    def open_method_b(request, *, timeout):
        messages = json.loads(request.data)["messages"]
        given = json.loads(messages[1]["content"])
        assert given == {
            "paragraph": PAIR.claim_text,
            "bibliography": PAIR.ref_raw,
            "citation_pairs": [{"pair_id": "pair_001", "claim": PAIR.claim_text, "ref_number": 1}],
        }
        b_requests.append(given)
        result = {"results": [{"pair_id": "pair_001", "support_score": 0.10}]}
        return BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(result)}}]}).encode())

    monkeypatch.setattr("src.method_b.urlopen", open_method_b)

    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.secrets["OPENROUTER_API_KEY"] = "test-placeholder"
    app.secrets["S2_API_KEY"] = "scholar-placeholder"
    app.secrets["OPENALEX_API_KEY"] = "openalex-placeholder"
    app.run()
    app.text_area[0].set_value(PAIR.claim_text)
    app.text_area[1].set_value(PAIR.ref_raw)
    app.toggle[0].set_value(False)
    app.button[1].click().run()
    assert not app.exception
    assert stub.events == sequence
    [record] = app.dataframe[0].value.to_dict("records")
    [row] = app.session_state["analysis_rows"]
    assert row.method_a.support_score == (None if choice == "unrelated" else 0.40)
    assert row.reference.abstract_source == provider
    assert record["方法 A 證據層級"] == band
    assert record["方法 A 判斷依據"] == basis
    assert record["來源狀態"] == source_status
    assert record["方法 A 摘要來源"] == {
        "semantic_scholar": "Semantic Scholar",
        "openalex": "OpenAlex",
        "": "",
    }[provider]
    if choice != "unrelated":
        expected_evidence = {
            "semantic_scholar": scholar_abstract,
            "openalex": openalex_abstract,
            "": TITLE.title,
        }
        assert stub.states[1]["evidence"] == expected_evidence[provider]
        assert record["方法 A 摘要節錄"] == (expected_evidence[provider] if provider else "")
        sent_title = stub.states[1].get("verified_source_title")
        audit = json.loads(record["方法 A 查核紀錄"])
        assert sent_title == (TITLE.title if provider else None)
        assert audit.get("verified_source_title") == sent_title
    assert record["方法 B 支持分數"] == 0.10
    assert record["方法 B 證據層級"] == "不支持"
    assert record["方法 B 判斷狀態"] == "完成"
    assert len(b_requests) == 1
    assert len(app.get("download_button")) == 1

    app.toggle[0].set_value(True).run()
    assert stub.events == sequence
    assert len(b_requests) == 1
    assert "使用 Jev" in app.info[0].value
