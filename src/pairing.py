"""Join extracted IEEE citations to the user's pasted reference list."""

from __future__ import annotations

from dataclasses import dataclass

from src.citation_parser import extract_claim_citations
from src.reference_parser import parse_reference_list
from src.schemas import CitationPair, ReferenceMatchStatus
from src.validation import IssueSeverity, ValidationIssue

# 汪老師 2026-11-17 流程圖第 ② 步：「每一篇被引用來源建立一筆查核項目」。
# 共引句拆成子主張時，原本對「每個引用 × 每個子主張」各產生一筆配對，兩篇來源會
# 變成四列，其中必有依設計就配錯的組合（[1] 的論文拿去判 [2] 的主張）。
# 預設一篇來源一列，判斷的陳述用原句；老師若確認要採用子主張拆解，改成 True 還原。
SPLIT_SHARED_SUBCLAIMS = False


@dataclass(frozen=True, slots=True)
class PairingOutcome:
    pairs: list[CitationPair]
    issues: list[ValidationIssue]


def build_citation_pairs(claim_text: str, reference_text: str) -> list[CitationPair]:
    """Create traceable pairs of claims and candidate cited sources."""

    return build_pairing_outcome(claim_text, reference_text).pairs


def build_pairing_outcome(claim_text: str, reference_text: str) -> PairingOutcome:
    """Return usable pairs plus non-fatal no-citation, duplicate and missing issues."""

    references = parse_reference_list(reference_text)
    pairs: list[CitationPair] = []
    issues: list[ValidationIssue] = []
    claims = extract_claim_citations(claim_text)

    if not claims:
        issues.append(
            ValidationIssue(
                code="NO_CITATIONS",
                message="段落中沒有偵測到可解析的 IEEE 數字引用。",
                severity=IssueSeverity.WARNING,
            )
        )

    for claim_number, claim in enumerate(claims, start=1):
        split_subclaims = SPLIT_SHARED_SUBCLAIMS and bool(claim.subclaims)
        evaluated_claims = claim.subclaims if split_subclaims else (claim.claim_text,)
        group_id = f"claim_{claim_number:03d}" if split_subclaims else ""
        # 不拆子主張時，原句就是判斷的陳述，不必再填 original_claim_text
        original_claim_text = "" if claim.subclaims and not split_subclaims else claim.original_claim_text
        for duplicate_number in claim.duplicate_numbers:
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_CITATION",
                    message=(
                        f"同一引用陳述重複出現 [{duplicate_number}]，"
                        "已合併為一筆查核項目。"
                    ),
                    severity=IssueSeverity.WARNING,
                )
            )
        for mention in claim.mentions:
            reference = references.get(mention.ref_number)
            if reference is None:
                status = ReferenceMatchStatus.MISSING_REFERENCE
                ref_raw = ""
                error_message = f"ref 清單找不到 [{mention.ref_number}]"
                issues.append(
                    ValidationIssue(
                        code="MISSING_REFERENCE",
                        message=error_message,
                        severity=IssueSeverity.WARNING,
                    )
                )
            else:
                status = ReferenceMatchStatus.MATCHED
                ref_raw = reference.raw_text
                error_message = ""

            for evaluated_claim in evaluated_claims:
                pairs.append(
                    CitationPair(
                        pair_id=f"pair_{len(pairs) + 1:03d}",
                        claim_text=evaluated_claim,
                        citation_marker=mention.marker,
                        ref_number=mention.ref_number,
                        ref_raw=ref_raw,
                        is_compound=claim.is_compound,
                        match_status=status,
                        error_message=error_message,
                        original_claim_text=original_claim_text,
                        claim_group_id=group_id,
                    )
                )
    return PairingOutcome(pairs=pairs, issues=issues)

