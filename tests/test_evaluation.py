"""Unit tests for the evaluation engine and metric computations."""

import pytest
from src.evaluation import BenchmarkQuestion, Evaluator, compute_answer_relevancy


def test_answer_relevancy_computation():
    question = "What are the core functions of the AMF in 5G architecture?"
    good_answer = "The AMF performs access control, mobility management, and N2 termination [TS 23.501 Clause 6.2.1, Page 423-424]."
    score = compute_answer_relevancy(question, good_answer)
    assert score >= 0.50

    abstention_answer = "I could not find sufficient supporting evidence in the indexed 3GPP documents."
    score_abstain = compute_answer_relevancy("What is the capital of France?", abstention_answer)
    assert score_abstain == 1.0


def test_evaluator_load_dataset():
    evaluator = Evaluator()
    questions = evaluator.load_benchmark_dataset()
    assert len(questions) == 25

    positives = [q for q in questions if not q.should_abstain]
    negatives = [q for q in questions if q.should_abstain]
    assert len(positives) == 21
    assert len(negatives) == 4


def test_evaluator_single_positive_query():
    evaluator = Evaluator()
    bq = BenchmarkQuestion(
        id="test_pos",
        question="What are the functions of the AMF in 5G architecture?",
        category="architecture",
        expected_document="3GPP TS 23.501",
        expected_clause="6.2.1, 4.2.2, 5.2.2",
        expected_page="423-424",
        should_abstain=False,
    )
    result = evaluator.evaluate_query(bq)
    assert result.is_abstained is False
    assert result.hit_rate == 1.0
    assert result.citation_precision >= 0.80
    assert result.abstention_correct is True


def test_evaluator_single_negative_query():
    evaluator = Evaluator()
    bq = BenchmarkQuestion(
        id="test_neg",
        question="What is the capital of France?",
        category="negative_abstention",
        should_abstain=True,
    )
    result = evaluator.evaluate_query(bq)
    assert result.is_abstained is True
    assert result.abstention_correct is True
    assert result.citation_precision == 1.0
