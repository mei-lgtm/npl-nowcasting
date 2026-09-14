"""News sentiment engine — economic/financial distress oriented.

Classical models (TF-IDF + LR/SVM) always run via scikit-learn.
Transformer names (IndoBERT, mBERT, XLM-R) load HuggingFace pipelines when
transformers+torch are installed; otherwise fall back to an Indo-trained
TF-IDF ensemble (still real sklearn inference, not a random stub).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.data.schemas.models import SentimentResult

log = logging.getLogger(__name__)

SECTORS = [
    "Manufacturing",
    "Mining",
    "Construction",
    "Property",
    "Trade",
    "Agriculture",
    "Transportation",
    "Consumer",
    "Financial Services",
    "Technology",
    "Energy",
]

# English + Indonesian distress cues (stress / NPL oriented)
NEGATIVE_CUES = [
    "default", "distress", "restructuring", "npl", "non-performing", "layoff",
    "bankruptcy", "slowdown", "weak", "deteriorat", "pressure", "impairment",
    "delinquen", "repayment pressure", "credit concern",
    "macet", "gagal bayar", "restrukturisasi", "kredit bermasalah", "phk",
    "pkpu", "pailit", "melemah", "merosot", "anjlok", "tekanan", "pencadangan",
    "beban bunga", "kesulitan", "merugi", "memburuk", "penurunan", "krisis",
    "default kredit", "nunggak", "kolektibilitas", "downgrade",
]
POSITIVE_CUES = [
    "improv", "recover", "strengthen", "stable demand", "lower restructuring",
    "repayment capacity", "earnings recovery",
    "membaik", "pulih", "stabil", "menurunnya npl", "kualitas kredit",
    "pelunasan", "tumbuh", "menguat", "surplus", "optimis", "reliabilitas",
    "penurunan npl", "npl turun", "coverage naik", "arus kas membaik",
]

SOURCE_WEIGHTS = {
    "DemoWire": 1.0,
    "EconDaily": 0.95,
    "BankingInsight": 1.1,
    "MacroPulse": 0.9,
    "CreditWatch": 1.15,
    "Google News": 1.05,
}

# Rich Indo+EN seed corpus so classical models actually learn domain language
_INDO_SEEDS: list[tuple[str, str]] = [
    ("Kredit macet sektor manufaktur naik tajam, bank perketat penyaluran", "negative"),
    ("Gelombang gagal bayar korporasi properti berlanjut kuartal ini", "negative"),
    ("Restrukturisasi kredit perdagangan melonjak dan NPL meningkat", "negative"),
    ("Pabrik tutup, ribuan pekerja terkena PHK di sektor industri", "negative"),
    ("Permohonan PKPU emiten tambang bertambah signifikan", "negative"),
    ("Penjualan lesu, arus kas debitur tertekan dan nunggak cicilan", "negative"),
    ("Bank besar naikkan pencadangan untuk portofolio kredit bermasalah", "negative"),
    ("Beban bunga naik, debitur kesulitan menyicil pinjaman modal kerja", "negative"),
    ("Harga komoditas jatuh, eksportir merugi dan kualitas kredit memburuk", "negative"),
    ("Rupiah melemah, utang valas perusahaan membengkak dan risiko default naik", "negative"),
    ("NPL bank umum naik ke level tertinggi dalam setahun", "negative"),
    ("OJK waspadai tekanan kredit di sektor konstruksi dan properti", "negative"),
    ("Downgrade rating emiten karena arus kas operasional melemah", "negative"),
    ("Rising defaults and restructuring pressure in corporate loans", "negative"),
    ("Corporate distress and layoffs increase NPL risk for banks", "negative"),
    ("Non-performing loans deteriorate amid repayment pressure", "negative"),
    ("Kualitas kredit membaik seiring pemulihan permintaan domestik", "positive"),
    ("Emiten lunasi utang lebih cepat dari jadwal, NPL turun", "positive"),
    ("Penyaluran kredit tumbuh dengan risiko terjaga dan coverage naik", "positive"),
    ("Marjin pulih, rasio utang menurun dan arus kas membaik", "positive"),
    ("Permintaan ekspor menguat, debitur menunjukkan kapasitas bayar lebih baik", "positive"),
    ("Bank turunkan pencadangan portofolio setelah kualitas kredit membaik", "positive"),
    ("NPL turun tipis didukung pemulihan sektor konsumen", "positive"),
    ("Borrowers show improved repayment capacity this quarter", "positive"),
    ("Credit quality improves with stronger economic activity", "positive"),
    ("Loan performance strengthens as restructuring stock declines", "positive"),
    ("Regulator terbitkan aturan baru pembiayaan sektor riil", "neutral"),
    ("Bank Indonesia pertahankan suku bunga acuan pada level saat ini", "neutral"),
    ("Asosiasi paparkan proyeksi pertumbuhan kredit tahun depan", "neutral"),
    ("OJK rilis data kinerja pembiayaan bulanan perbankan", "neutral"),
    ("Credit quality remains broadly stable month to month", "neutral"),
    ("Loan performance little changed this month across sectors", "neutral"),
    ("BI Rate tetap, pasar menunggu data inflasi berikutnya", "neutral"),
    ("Statistik perbankan menunjukkan penyaluran kredit yang flat", "neutral"),
]

TRANSFORMER_ALIASES = {
    "indobert": "indobenchmark/indobert-base-p1",
    "mbert": "nlptown/bert-base-multilingual-uncased-sentiment",
    "xlm_roberta": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}


def _infer_sector(text: str) -> Optional[str]:
    lower = text.lower()
    mapping = {
        "manufaktur": "Manufacturing", "industri": "Manufacturing",
        "tambang": "Mining", "mining": "Mining",
        "konstruksi": "Construction", "properti": "Property", "property": "Property",
        "perdagangan": "Trade", "retail": "Trade",
        "pertanian": "Agriculture", "transport": "Transportation",
        "konsumen": "Consumer", "bank": "Financial Services", "keuangan": "Financial Services",
        "teknologi": "Technology", "energi": "Energy", "migas": "Energy",
    }
    for k, v in mapping.items():
        if k in lower:
            return v
    for s in SECTORS:
        if s.lower() in lower:
            return s
    return None


def _rule_scores(text: str) -> dict:
    lower = text.lower()
    neg = sum(1 for c in NEGATIVE_CUES if c in lower)
    pos = sum(1 for c in POSITIVE_CUES if c in lower)
    score = (neg - pos) / max(neg + pos, 1)
    sentiment_score = float(np.clip(0.5 + 0.45 * score, 0.05, 0.95))
    if score > 0.2:
        sentiment = "negative"
        credit = "deteriorating"
        distress = "high" if score > 0.5 else "medium"
        econ = "high" if score > 0.45 else "medium"
        bank = "medium" if score > 0.3 else "low"
    elif score < -0.2:
        sentiment = "positive"
        credit = "improving"
        distress = "low"
        econ = "low"
        bank = "low"
    else:
        sentiment = "neutral"
        credit = "stable"
        distress = "medium"
        econ = "low"
        bank = "low"
    relevance = float(
        np.clip(
            0.4
            + 0.08 * (neg + pos)
            + (0.2 if re.search(r"\bnpl\b|kredit|credit|macet|default", lower) else 0),
            0.2,
            0.98,
        )
    )
    return {
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "credit_risk": credit,
        "financial_distress": distress,
        "economic_stress": econ,
        "banking_risk": bank,
        "npl_relevance": relevance,
    }


def _map_transformer_label(label: str, score: float) -> tuple[str, float]:
    lab = (label or "").lower()
    # nlptown: LABEL_1..LABEL_5 (1=neg … 5=pos)
    if lab.startswith("label_"):
        try:
            stars = int(lab.split("_")[1])
        except Exception:
            stars = 3
        if stars <= 2:
            return "negative", float(0.55 + 0.35 * score)
        if stars >= 4:
            return "positive", float(0.45 - 0.35 * score)
        return "neutral", 0.5
    if any(x in lab for x in ("neg", "bad", "LABEL_0")):
        return "negative", float(0.55 + 0.4 * score)
    if any(x in lab for x in ("pos", "good", "LABEL_2")):
        return "positive", float(0.45 - 0.4 * score)
    return "neutral", 0.5


class SentimentEngine:
    """Selectable sentiment models with distress-oriented outputs."""

    AVAILABLE = [
        "tfidf_logistic",
        "tfidf_svm",
        "indobert",
        "mbert",
        "xlm_roberta",
        "llm_zero_shot",
        "llm_structured",
        "ensemble",
        "rule_based",
    ]

    _hf_pipes: dict = {}

    def __init__(self):
        self._classical: dict = {}
        self._fitted = False

    def fit_classical(self, texts: list[str], labels: list[str]):
        y = np.array(labels)
        self._classical["tfidf_logistic"] = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        self._classical["tfidf_svm"] = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)),
                ("clf", LinearSVC(class_weight="balanced", max_iter=4000)),
            ]
        )
        for pipe in self._classical.values():
            pipe.fit(texts, y)
        self._fitted = True

    def _ensure_fitted(self):
        if self._fitted:
            return
        texts = [t for t, _ in _INDO_SEEDS]
        labels = [l for _, l in _INDO_SEEDS]
        self.fit_classical(texts, labels)

    def _try_transformer(self, model: str, text: str) -> Optional[dict]:
        """Run HF pipeline if transformers+torch available."""
        if model not in TRANSFORMER_ALIASES:
            return None
        try:
            from transformers import pipeline  # type: ignore
        except Exception:
            return None
        if model not in SentimentEngine._hf_pipes:
            repo = TRANSFORMER_ALIASES[model]
            try:
                log.info("Loading transformer sentiment model %s (%s)", model, repo)
                SentimentEngine._hf_pipes[model] = pipeline(
                    "sentiment-analysis",
                    model=repo,
                    truncation=True,
                    max_length=256,
                )
            except Exception as e:
                log.warning("Failed to load %s: %s", model, e)
                SentimentEngine._hf_pipes[model] = None
        pipe = SentimentEngine._hf_pipes.get(model)
        if pipe is None:
            return None
        try:
            out = pipe(text[:500])[0]
            sentiment, sscore = _map_transformer_label(out.get("label", ""), float(out.get("score", 0.5)))
            rules = _rule_scores(text)
            rules["sentiment"] = sentiment
            rules["sentiment_score"] = float(np.clip(sscore, 0.05, 0.95))
            rules["model_used"] = model
            return rules
        except Exception as e:
            log.warning("Transformer inference failed (%s): %s", model, e)
            return None

    def _classical_predict(self, text: str, model: str) -> dict:
        self._ensure_fitted()
        rules = _rule_scores(text)
        pipe = self._classical.get(model)
        if pipe is None:
            rules["model_used"] = "rule_based"
            return rules
        pred = pipe.predict([text])[0]
        rules["sentiment"] = pred
        # Blend rule score with classifier direction
        if pred == "negative":
            rules["sentiment_score"] = float(max(rules["sentiment_score"], 0.62))
            try:
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    proba = pipe.predict_proba([text])[0]
                    classes = list(pipe.named_steps["clf"].classes_)
                    if "negative" in classes:
                        rules["sentiment_score"] = float(
                            0.35 + 0.6 * proba[classes.index("negative")]
                        )
            except Exception:
                pass
        elif pred == "positive":
            rules["sentiment_score"] = float(min(rules["sentiment_score"], 0.38))
        rules["model_used"] = model
        return rules

    def analyze(self, text: str, model: str = "tfidf_logistic", source: str = "DemoWire") -> SentimentResult:
        model = model or "tfidf_logistic"
        text = (text or "").strip() or " "

        if model in ("indobert", "mbert", "xlm_roberta"):
            result = self._try_transformer(model, text)
            if result is None:
                # Real classical Indo-trained fallback (not random)
                result = self._classical_predict(text, "tfidf_logistic")
                result["model_used"] = f"{model}_fallback_tfidf_indo"
        elif model in ("llm_zero_shot", "llm_structured"):
            # No API key path: use ensemble of rules + classical
            self._ensure_fitted()
            rules = _rule_scores(text)
            votes = [rules["sentiment"]]
            for name in ("tfidf_logistic", "tfidf_svm"):
                votes.append(self._classical[name].predict([text])[0])
            sentiment = max(set(votes), key=votes.count)
            result = rules
            result["sentiment"] = sentiment
            if sentiment == "negative":
                result["sentiment_score"] = max(result["sentiment_score"], 0.6)
            elif sentiment == "positive":
                result["sentiment_score"] = min(result["sentiment_score"], 0.4)
            result["model_used"] = f"{model}_fallback_ensemble"
        elif model == "ensemble":
            self._ensure_fitted()
            rules = _rule_scores(text)
            votes = [rules["sentiment"]]
            for name in ("tfidf_logistic", "tfidf_svm"):
                votes.append(self._classical[name].predict([text])[0])
            sentiment = max(set(votes), key=votes.count)
            result = rules
            result["sentiment"] = sentiment
            result["model_used"] = "ensemble"
        elif model in ("tfidf_logistic", "tfidf_svm"):
            result = self._classical_predict(text, model)
        else:
            result = _rule_scores(text)
            result["model_used"] = "rule_based"

        sector = _infer_sector(text)
        sw = SOURCE_WEIGHTS.get(source, 1.0)
        signed = result["sentiment_score"] * 2 - 1
        if result["sentiment"] == "positive":
            signed = -abs(signed)
        elif result["sentiment"] == "negative":
            signed = abs(signed)
        else:
            signed = 0.0
        stress = float(np.clip(signed * result["npl_relevance"] * sw, -1, 1))

        return SentimentResult(
            sentiment=result["sentiment"],
            sentiment_score=float(result["sentiment_score"]),
            credit_risk=result["credit_risk"],
            financial_distress=result["financial_distress"],
            economic_stress=result["economic_stress"],
            banking_risk=result["banking_risk"],
            sector=sector,
            npl_relevance=float(result["npl_relevance"]),
            stress_contribution=stress,
            model_used=result["model_used"],
        )

    def analyze_batch(
        self,
        articles: list[dict],
        model: str = "tfidf_logistic",
    ) -> list[dict]:
        """Score many headlines; returns list aligned with input."""
        out = []
        for a in articles:
            text = a.get("headline") or a.get("text") or a.get("article_text") or ""
            source = a.get("source") or "DemoWire"
            r = self.analyze(text, model=model, source=source)
            out.append(
                {
                    "id": a.get("id"),
                    "headline": text,
                    "sentiment": r.sentiment,
                    "sentiment_score": r.sentiment_score,
                    "stress_contribution": r.stress_contribution,
                    "npl_relevance": r.npl_relevance,
                    "sector": r.sector or a.get("sector"),
                    "model_used": r.model_used,
                    "label_id": "negatif"
                    if r.sentiment == "negative"
                    else ("positif" if r.sentiment == "positive" else "netral"),
                    # SPA News Stress convention: + = stress (bad news)
                    "base": float(r.stress_contribution),
                    "rel": float(r.npl_relevance),
                    "neg": r.sentiment == "negative",
                    "pos": r.sentiment == "positive",
                }
            )
        return out

    def news_stress_index(
        self,
        articles: list[dict],
        model: str = "tfidf_logistic",
        as_of=None,
    ) -> dict:
        if not articles:
            return {
                "news_stress_index": 0.0,
                "article_count": 0,
                "negative_ratio": 0.0,
                "avg_sentiment": 0.5,
                "top_topics": [],
                "model_used": model,
            }

        scored = self.analyze_batch(articles, model=model)
        scores = [s["stress_contribution"] for s in scored]
        neg = sum(1 for s in scored if s["sentiment"] == "negative")
        topics: dict[str, int] = {}
        for a, s in zip(articles, scored):
            topic = a.get("topic") or s.get("sector") or "general"
            topics[topic] = topics.get(topic, 0) + 1
        top = sorted(topics.items(), key=lambda x: -x[1])[:5]
        return {
            "news_stress_index": float(np.clip(np.mean(scores), -1, 1)),
            "article_count": len(articles),
            "negative_ratio": neg / len(articles),
            "avg_sentiment": float(np.mean([s["sentiment_score"] for s in scored])),
            "top_topics": [{"topic": t, "count": c} for t, c in top],
            "model_used": scored[0]["model_used"] if scored else model,
            "articles": scored,
        }
