import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from src.adapters import ReferenceLookupResult
from src.method_b import CHAT_URL, DEFAULT_METHOD_B_MODEL, UNVERIFIABLE_STATUS, run_method_b
from src.pairing import build_citation_pairs
from src.schemas import AnalysisRow, EvidenceBasis, MethodResult, SourceStatus, SupportBand


PARAGRAPH = "Alpha was published in 2020 [1]. Beta was published in 2020 [2]."
REFS = '[1] "Alpha," 2020.\n[2] "Beta," 2019.'


def abstract_of(pair):
    return f"Abstract of reference {pair.ref_number}."


def with_abstract(pair, method_a=None):
    reference = ReferenceLookupResult(pair.ref_number, "title", abstract=abstract_of(pair),
                                      source_status=SourceStatus.GOT_ABSTRACT)
    return AnalysisRow(pair, method_a=method_a, reference=reference)


def response(answers):
    body = {
        "id": "test-comparison",
        "model": "test-model",
        "choices": [{"message": {"content": json.dumps({"results": answers})}}],
    }
    return BytesIO(json.dumps(body).encode())


def test_method_b_attaches_model_scores_by_id_without_overriding_the_band(monkeypatch):
    paragraph = PARAGRAPH + " Beta improved accuracy by 10% [2]. Alpha outperforms Beta [1]."
    rows = [with_abstract(pair, MethodResult(0.20, EvidenceBasis.ABSTRACT))
            for pair in build_citation_pairs(paragraph, REFS)]

    def open_response(request, *, timeout):
        assert request.full_url == CHAT_URL
        payload = json.loads(request.data)
        assert payload["plugins"] == [{"id": "web", "enabled": False}]
        assert "tools" not in payload
        supplied = json.loads(payload["messages"][1]["content"])
        assert set(supplied) == {"paragraph", "citation_pairs"}
        assert supplied["paragraph"] == paragraph
        assert supplied["citation_pairs"] == [
            {"pair_id": row.pair.pair_id, "claim": row.pair.claim_text,
             "cited_as": row.pair.citation_marker, "evidence": abstract_of(row.pair)}
            for row in rows
        ]
        assert payload["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"][
            "required"
        ] == ["pair_id", "support_score"]
        return response([
            {"pair_id": rows[3].pair.pair_id, "support_score": 0.91},
            {"pair_id": rows[2].pair.pair_id, "support_score": 0.55},
            {"pair_id": rows[1].pair.pair_id, "support_score": 0.10},
            {"pair_id": rows[0].pair.pair_id, "support_score": 0.80},
        ])

    monkeypatch.setattr("src.method_b.urlopen", open_response)
    compared = run_method_b(rows, paragraph, REFS, "placeholder")
    assert [row.pair for row in compared] == [row.pair for row in rows]
    assert [row.method_a for row in compared] == [row.method_a for row in rows]
    assert [row.method_b.support_score for row in compared] == [0.80, 0.10, 0.55, 0.91]
    assert all(row.method_b.evidence_basis is EvidenceBasis.ABSTRACT for row in compared)
    assert [row.method_b.band for row in compared] == [
        SupportBand.SUPPORTED, SupportBand.UNSUPPORTED, SupportBand.UNCERTAIN, SupportBand.SUPPORTED,
    ]
    assert [row.direct_judgment.status for row in compared] == ["完成"] * 4
    audit = json.loads(compared[1].to_record()["方法 B 查核紀錄"])
    assert audit == {"model": "test-model", "request_id": "test-comparison", "raw_score": 0.10}


@pytest.mark.parametrize("model", [f"{DEFAULT_METHOD_B_MODEL}:online", "perplexity/sonar"])
def test_unverified_model_does_not_send_a_request(monkeypatch, model):
    [pair] = build_citation_pairs("Alpha was published in 2020 [1].", REFS)
    row = with_abstract(pair, MethodResult(0.8, EvidenceBasis.ABSTRACT))

    def unexpected_request(*args, **kwargs):
        pytest.fail("An unverified model must not send an API request")

    monkeypatch.setattr("src.method_b.urlopen", unexpected_request)
    [compared] = run_method_b([row], pair.claim_text, REFS, "placeholder", model=model)
    assert compared.method_a == row.method_a
    assert compared.method_b is None
    assert "GPT-4.1 Mini" in compared.direct_judgment.status


@pytest.mark.parametrize(("message_details", "usage"), [
    ({"annotations": [{"type": "url_citation"}]}, {}),
    ({"tool_calls": [{"type": "function"}]}, {}),
    ({}, {"server_tool_use": {"web_search_requests": 1}}),
    ({}, {"server_tool_use": {"web_fetch_requests": 1}}),
])
def test_response_reporting_external_tools_is_not_scored(monkeypatch, message_details, usage):
    [pair] = build_citation_pairs("Alpha was published in 2020 [1].", REFS)
    row = with_abstract(pair, MethodResult(0.80, EvidenceBasis.ABSTRACT))
    result = {"results": [{"pair_id": pair.pair_id, "support_score": 0.98}]}
    body = {"usage": usage, "choices": [{"message": {
        "content": json.dumps(result), **message_details,
    }}]}
    monkeypatch.setattr("src.method_b.urlopen", lambda *args, **kwargs: BytesIO(json.dumps(body).encode()))
    [compared] = run_method_b([row], pair.claim_text, REFS, "placeholder")
    assert compared.method_a == row.method_a
    assert compared.method_b is None
    assert "外部工具" in compared.direct_judgment.status


def test_api_failure_preserves_method_a_and_leaves_b_ungraded(monkeypatch):
    [pair] = build_citation_pairs("Alpha was published in 2020 [1].", REFS)
    row = with_abstract(pair, MethodResult(0.80, EvidenceBasis.ABSTRACT))

    def reject(request, *, timeout):
        raise HTTPError(CHAT_URL, 429, "Too many requests", None, BytesIO(b"private upstream message"))

    monkeypatch.setattr("src.method_b.urlopen", reject)
    [compared] = run_method_b([row], pair.claim_text, REFS, "placeholder")
    assert compared.method_a == row.method_a
    assert compared.method_b is None
    assert compared.direct_judgment.status == "OpenRouter 請求失敗（HTTP 429）"


def test_missing_reference_and_duplicate_decisions_are_not_assigned_scores(monkeypatch):
    paragraph = PARAGRAPH + " Gamma was published in 2020 [3]."
    rows = [with_abstract(pair) for pair in build_citation_pairs(paragraph, REFS)]

    def open_response(request, *, timeout):
        payload = json.loads(request.data)
        supplied = json.loads(payload["messages"][1]["content"])
        assert [pair["cited_as"] for pair in supplied["citation_pairs"]] == ["[1]", "[2]"]
        repeated = {"pair_id": rows[0].pair.pair_id, "support_score": 0.9}
        return response([repeated, repeated, {"pair_id": "unexpected"}])

    monkeypatch.setattr("src.method_b.urlopen", open_response)
    compared = run_method_b(rows, paragraph, REFS, "placeholder")
    assert all(row.method_b is None for row in compared)
    assert [row.direct_judgment.status for row in compared] == [
        "回覆包含重複配對，暫不評分", "回覆缺少此配對，暫不評分", "缺少 ref，暫不評分",
    ]


def test_pairs_without_abstract_are_unverifiable_and_not_sent(monkeypatch):
    pairs = build_citation_pairs(PARAGRAPH, REFS)
    no_abstract = ReferenceLookupResult(pairs[1].ref_number, "Beta", source_status=SourceStatus.NOT_FOUND)
    rows = [with_abstract(pairs[0]), AnalysisRow(pairs[1], reference=no_abstract)]

    def open_response(request, *, timeout):
        supplied = json.loads(json.loads(request.data)["messages"][1]["content"])
        assert [pair["pair_id"] for pair in supplied["citation_pairs"]] == [pairs[0].pair_id]
        return response([{"pair_id": pairs[0].pair_id, "support_score": 0.75}])

    monkeypatch.setattr("src.method_b.urlopen", open_response)
    compared = run_method_b(rows, PARAGRAPH, REFS, "placeholder")
    assert compared[0].method_b.band is SupportBand.SUPPORTED
    assert compared[1].method_b.support_score is None
    assert compared[1].method_b.band is SupportBand.UNVERIFIABLE
    assert compared[1].direct_judgment.status == UNVERIFIABLE_STATUS


def test_no_evidence_at_all_sends_no_request(monkeypatch):
    rows = [AnalysisRow(pair) for pair in build_citation_pairs(PARAGRAPH, REFS)]
    monkeypatch.setattr("src.method_b.urlopen", lambda *a, **k: pytest.fail("no evidence, no request"))
    compared = run_method_b(rows, PARAGRAPH, REFS, "placeholder")
    assert [row.method_b.band for row in compared] == [SupportBand.UNVERIFIABLE] * 2


@pytest.mark.parametrize("found", [True, False])
def test_title_stopped_pairs_are_looked_up_for_method_b(monkeypatch, found):
    [pair] = build_citation_pairs("Alpha was published in 2020 [1].", REFS)
    skipped = ReferenceLookupResult(pair.ref_number, "Alpha", source_status=SourceStatus.SKIPPED)
    row = AnalysisRow(pair, reference=skipped)
    looked_up = []

    def lookup(given):
        looked_up.append(given.pair_id)
        if not found:
            raise TimeoutError()
        return ReferenceLookupResult(given.ref_number, "Alpha", abstract="Alpha abstract.",
                                     source_status=SourceStatus.GOT_ABSTRACT)

    def open_response(request, *, timeout):
        supplied = json.loads(json.loads(request.data)["messages"][1]["content"])
        assert supplied["citation_pairs"][0]["evidence"] == "Alpha abstract."
        return response([{"pair_id": pair.pair_id, "support_score": 0.2}])

    monkeypatch.setattr("src.method_b.urlopen", open_response)
    [compared] = run_method_b([row], pair.claim_text, REFS, "placeholder", evidence_lookup=lookup)
    assert looked_up == [pair.pair_id]
    assert compared.reference == skipped  # method A's record is untouched
    assert compared.method_b.band is (SupportBand.UNSUPPORTED if found else SupportBand.UNVERIFIABLE)
