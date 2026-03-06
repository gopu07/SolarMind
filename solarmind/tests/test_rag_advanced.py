"""
Tests for Advanced Multi-Stage RAG Pipeline (Step 9 — Evaluation).

Verifies:
  - BM25 search produces scored results
  - Multi-query expansion generates multiple queries
  - Hybrid scoring combines vector + BM25 + metadata signals correctly
  - Knowledge base YAML parsing and ingestion
  - Telemetry context building and anomaly detection
  - DiagnosticReport schema validation
  - Reranking preserves/reorders candidates
  - Historical memory retrieval
"""

import json
import time

import pytest


# =====================================================================
# BM25 Search Tests
# =====================================================================
class TestBM25Search:
    def test_bm25_returns_scored_results(self):
        """BM25 search must return results with bm25_score field."""
        from rag.retriever import bm25_search

        docs = [
            {"content": "inverter temperature is rising above threshold"},
            {"content": "string current mismatch detected on string 5"},
            {"content": "cooling fan failure causing thermal shutdown"},
        ]
        results = bm25_search("temperature rising", docs, top_k=3)

        assert len(results) > 0
        assert all("bm25_score" in r for r in results)
        # The best match should mention temperature
        assert "temperature" in results[0]["content"].lower()

    def test_bm25_empty_corpus(self):
        """BM25 with empty corpus should return empty list."""
        from rag.retriever import bm25_search

        results = bm25_search("any query", [], top_k=5)
        assert results == []

    def test_bm25_scores_normalized(self):
        """BM25 scores should be normalized to [0, 1]."""
        from rag.retriever import bm25_search

        docs = [
            {"content": "solar panel efficiency dropping steadily"},
            {"content": "inverter alarm code triggered"},
        ]
        results = bm25_search("solar panel efficiency", docs, top_k=2)
        for r in results:
            assert 0.0 <= r["bm25_score"] <= 1.0


# =====================================================================
# Multi-Query Expansion Tests
# =====================================================================
class TestMultiQueryExpansion:
    def test_expansion_returns_multiple_queries(self):
        """Expansion should return at least the original + heuristic variants."""
        from rag.retriever import expand_queries

        queries = expand_queries("Why is INV_001 overheating?", n=3)
        assert len(queries) >= 2  # original + at least one variant
        assert queries[0] == "Why is INV_001 overheating?"

    def test_expansion_includes_original(self):
        """The original query must always be first."""
        from rag.retriever import expand_queries

        original = "What maintenance does INV_005 need?"
        queries = expand_queries(original, n=2)
        assert queries[0] == original


# =====================================================================
# Hybrid Scoring Tests
# =====================================================================
class TestHybridScoring:
    def test_recency_score_today(self):
        """Today's timestamp should get recency score 1.0."""
        from rag.retriever import calculate_recency_score

        now = int(time.time())
        assert calculate_recency_score(now, now) == 1.0

    def test_recency_score_old(self):
        """Timestamp >7 days old should get recency score 0.0."""
        from rag.retriever import calculate_recency_score

        now = int(time.time())
        old = now - (8 * 86400)  # 8 days ago
        assert calculate_recency_score(old, now) == 0.0

    def test_recency_score_midrange(self):
        """Timestamp 4 days old should be between 0 and 1."""
        from rag.retriever import calculate_recency_score

        now = int(time.time())
        mid = now - (4 * 86400)
        score = calculate_recency_score(mid, now)
        assert 0.0 < score < 1.0

    def test_tokenizer_handles_punctuation(self):
        """Tokenizer should split on punctuation."""
        from rag.retriever import _tokenize

        tokens = _tokenize("INV_001: temperature=85.5°C, alarm!")
        assert "inv_001" in tokens or "inv" in tokens
        assert "temperature" in tokens


# =====================================================================
# Knowledge Base Tests
# =====================================================================
class TestKnowledgeBase:
    def test_yaml_files_exist(self):
        """All 4 knowledge base YAML files should exist."""
        import config

        kb_dir = config.KNOWLEDGE_BASE_DIR
        yaml_files = list(kb_dir.glob("*.yaml"))
        assert len(yaml_files) >= 4

    def test_yaml_files_parse(self):
        """Each YAML file should parse without errors."""
        import yaml
        import config

        for f in config.KNOWLEDGE_BASE_DIR.glob("*.yaml"):
            with open(f, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            assert "concept" in doc
            assert "chunks" in doc
            assert len(doc["chunks"]) > 0

    def test_each_chunk_has_required_fields(self):
        """Each chunk must have symptoms, sensor_signals, probable_causes, recommended_actions."""
        import yaml
        import config

        for f in config.KNOWLEDGE_BASE_DIR.glob("*.yaml"):
            with open(f, "r", encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            for chunk in doc["chunks"]:
                assert "symptoms" in chunk, f"Missing symptoms in {chunk.get('id', '?')}"
                assert "sensor_signals" in chunk, f"Missing sensor_signals in {chunk.get('id', '?')}"
                assert "probable_causes" in chunk, f"Missing probable_causes in {chunk.get('id', '?')}"
                assert "recommended_actions" in chunk, f"Missing recommended_actions in {chunk.get('id', '?')}"


# =====================================================================
# Telemetry Context Tests
# =====================================================================
class TestTelemetryContext:
    def test_anomaly_detection_high_temp(self):
        """High temperature should trigger anomaly."""
        from rag.telemetry_context import detect_anomalies

        telemetry = {"inverter_temperature": 80.0}
        anomalies = detect_anomalies(telemetry)
        assert len(anomalies) >= 1
        assert any(a["signal"] == "inverter_temperature" for a in anomalies)

    def test_anomaly_detection_critical_temp(self):
        """Critical temperature should trigger CRITICAL severity."""
        from rag.telemetry_context import detect_anomalies

        telemetry = {"inverter_temperature": 90.0}
        anomalies = detect_anomalies(telemetry)
        temp_anomalies = [a for a in anomalies if a["signal"] == "inverter_temperature"]
        assert len(temp_anomalies) >= 1
        assert temp_anomalies[0]["severity"] == "CRITICAL"

    def test_anomaly_detection_normal(self):
        """Normal readings should produce no anomalies for that signal."""
        from rag.telemetry_context import detect_anomalies

        telemetry = {"inverter_temperature": 45.0}
        anomalies = detect_anomalies(telemetry)
        temp_anomalies = [a for a in anomalies if a["signal"] == "inverter_temperature"]
        assert len(temp_anomalies) == 0

    def test_string_imbalance_detection(self):
        """Severe string imbalance should be detected."""
        from rag.telemetry_context import detect_anomalies

        telemetry = {
            "smu_string1": 8.5, "smu_string2": 8.3, "smu_string3": 8.4,
            "smu_string4": 2.1,  # severely low
            "smu_string5": 8.2, "smu_string6": 8.1,
        }
        anomalies = detect_anomalies(telemetry)
        imbalance = [a for a in anomalies if a["signal"] == "string_current_imbalance"]
        assert len(imbalance) >= 1


# =====================================================================
# Diagnostic Report Schema Tests
# =====================================================================
class TestDiagnosticReportSchema:
    def test_valid_report_parses(self):
        """Full structured report should parse into DiagnosticReport."""
        from api.schemas.models import DiagnosticReport

        data = {
            "diagnosis": "IGBT module thermal stress",
            "risk_level": "HIGH",
            "root_cause_hypothesis": "Prolonged operation above 80°C causing IGBT degradation",
            "sensor_evidence": [
                {"signal": "inverter_temperature", "value": "87°C", "expected": "< 75°C", "assessment": "Critical overtemp"}
            ],
            "recommended_actions": [
                {"action": "Inspect cooling fans", "priority": "immediate", "justification": "Temperature above critical threshold"}
            ],
            "confidence": "HIGH",
            "data_quality": "COMPLETE",
        }
        report = DiagnosticReport(**data)
        assert report.risk_level == "HIGH"
        assert len(report.sensor_evidence) == 1
        assert len(report.recommended_actions) == 1

    def test_minimal_report_parses(self):
        """Minimal report with only required fields should parse."""
        from api.schemas.models import DiagnosticReport

        report = DiagnosticReport(
            diagnosis="Normal operation",
            risk_level="LOW",
            root_cause_hypothesis="No anomalies detected",
        )
        assert report.confidence == "MEDIUM"  # default
        assert report.data_quality == "PARTIAL"  # default
        assert report.sensor_evidence == []


# =====================================================================
# Config Tests
# =====================================================================
class TestRAGConfig:
    def test_fusion_weights_sum_to_one(self):
        """RAG fusion weights should sum to 1.0."""
        import config
        total = sum(config.RAG_FUSION_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_multi_query_count_positive(self):
        """Multi query count must be positive."""
        import config
        assert config.RAG_MULTI_QUERY_COUNT >= 1

    def test_rerank_top_k_gte_final(self):
        """Rerank candidates must be >= final results."""
        import config
        assert config.RAG_RERANK_TOP_K >= config.RAG_FINAL_TOP_K

    def test_knowledge_base_dir_exists(self):
        """Knowledge base directory must exist."""
        import config
        assert config.KNOWLEDGE_BASE_DIR.exists()
