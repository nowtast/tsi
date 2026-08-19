"""Structural and displayed-mathematics parity checks for paired TeX packages."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import re


_INPUT_PATTERN = re.compile(r"\\input\{sections/([^}]+)\}")
_ENVIRONMENT_PATTERN = re.compile(
    r"\\begin\{(definition|assumption|lemma|proposition|theorem|corollary)\}"
)
_CITATION_PATTERN = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
_LABEL_PATTERN = re.compile(r"\\label\{([^}]+)\}")
_REFERENCE_PATTERN = re.compile(r"\\(?:eqref|ref)\{([^}]+)\}")
_DISPLAY_PATTERN = re.compile(
    r"""
    \\\[(?P<bracket>.*?)\\\]
    |
    \\begin\{(?P<environment>
        align\*?|equation\*?|gather\*?|multline\*?
    )\}(?P<body>.*?)\\end\{(?P=environment)\}
    """,
    re.DOTALL | re.VERBOSE,
)
_COMMENT_PATTERN = re.compile(r"(?<!\\)%[^\n]*")
_TRANSLATABLE_TEXT_PATTERN = re.compile(r"\\text\{[^{}]*\}")
_DECIMAL_PATTERN = re.compile(r"(?<![A-Za-z])\d+\.\d+")


@dataclass(frozen=True)
class ParityError:
    paper: str
    location: str
    message: str


def _inputs(package: Path) -> list[str]:
    return _INPUT_PATTERN.findall((package / "main.tex").read_text())


def _formal_environments(path: Path) -> list[str]:
    return _ENVIRONMENT_PATTERN.findall(path.read_text())


def _citation_keys(path: Path) -> set[str]:
    return {
        key.strip()
        for group in _CITATION_PATTERN.findall(path.read_text())
        for key in group.split(",")
    }


def _labels(path: Path) -> list[str]:
    return _LABEL_PATTERN.findall(path.read_text())


def _references(path: Path) -> list[str]:
    return _REFERENCE_PATTERN.findall(path.read_text())


def _display_math(path: Path) -> list[str]:
    """Return whitespace-insensitive display formulas in source order."""

    source = _COMMENT_PATTERN.sub("", path.read_text())
    formulas: list[str] = []
    for match in _DISPLAY_PATTERN.finditer(source):
        body = match.group("bracket")
        if body is None:
            body = match.group("body")
        body = _TRANSLATABLE_TEXT_PATTERN.sub(r"\\text{#}", body)
        formulas.append(re.sub(r"\s+", "", body).rstrip(",.;"))
    return formulas


def _decimal_literals(path: Path) -> Counter[str]:
    source = _COMMENT_PATTERN.sub("", path.read_text())
    return Counter(_DECIMAL_PATTERN.findall(source))


def _paired_papers(papers_root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in papers_root.iterdir()
            if path.is_dir()
            and not path.name.endswith("_ko")
            and (papers_root / f"{path.name}_ko").is_dir()
        )
    )


def check_bilingual_parity(
    repository_root: Path,
    papers: tuple[str, ...] | None = None,
) -> list[ParityError]:
    """Return structural or displayed-mathematics drift in bilingual packages."""

    errors: list[ParityError] = []
    papers_root = repository_root / "papers"
    selected_papers = papers if papers is not None else _paired_papers(papers_root)
    for paper in selected_papers:
        english = papers_root / paper
        korean = papers_root / f"{paper}_ko"
        english_inputs = _inputs(english)
        korean_inputs = _inputs(korean)
        if english_inputs != korean_inputs:
            errors.append(
                ParityError(
                    paper,
                    "main.tex",
                    f"section input order differs: {english_inputs!r} != {korean_inputs!r}",
                )
            )
            continue

        for section in english_inputs:
            english_path = english / "sections" / f"{section}.tex"
            korean_path = korean / "sections" / f"{section}.tex"
            if not english_path.exists() or not korean_path.exists():
                errors.append(ParityError(paper, section, "paired section file is missing"))
                continue

            english_envs = _formal_environments(english_path)
            korean_envs = _formal_environments(korean_path)
            if english_envs != korean_envs:
                errors.append(
                    ParityError(
                        paper,
                        section,
                        f"formal environment order differs: {english_envs!r} != {korean_envs!r}",
                    )
                )

            english_labels = _labels(english_path)
            korean_labels = _labels(korean_path)
            if english_labels != korean_labels:
                errors.append(
                    ParityError(
                        paper,
                        section,
                        f"label order differs: {english_labels!r} != {korean_labels!r}",
                    )
                )

            english_references = _references(english_path)
            korean_references = _references(korean_path)
            if english_references != korean_references:
                errors.append(
                    ParityError(
                        paper,
                        section,
                        "cross-reference order differs: "
                        f"{english_references!r} != {korean_references!r}",
                    )
                )

            english_citations = _citation_keys(english_path)
            korean_citations = _citation_keys(korean_path)
            if english_citations != korean_citations:
                errors.append(
                    ParityError(
                        paper,
                        section,
                        "citation keys differ: "
                        f"English-only={sorted(english_citations - korean_citations)!r}, "
                        f"Korean-only={sorted(korean_citations - english_citations)!r}",
                    )
                )

            english_math = _display_math(english_path)
            korean_math = _display_math(korean_path)
            if english_math != korean_math:
                first_difference = next(
                    (
                        index
                        for index in range(min(len(english_math), len(korean_math)))
                        if english_math[index] != korean_math[index]
                    ),
                    min(len(english_math), len(korean_math)),
                )
                errors.append(
                    ParityError(
                        paper,
                        section,
                        "display-math sequence differs "
                        f"at index {first_difference}: "
                        f"{len(english_math)} English vs {len(korean_math)} Korean",
                    )
                )

            english_decimals = _decimal_literals(english_path)
            korean_decimals = _decimal_literals(korean_path)
            if english_decimals != korean_decimals:
                errors.append(
                    ParityError(
                        paper,
                        section,
                        "decimal literals differ: "
                        f"English-only={english_decimals - korean_decimals!r}, "
                        f"Korean-only={korean_decimals - english_decimals!r}",
                    )
                )
    return errors
