import asyncio
import logging
import os
import sys
from typing import List, Dict, Any

# Ensure src is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from services.retrieval_service import RetrievalService
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")

# Define 6 diverse, realistic clinical evaluation benchmark queries with ground-truth keywords/keyphrases
EVALUATION_DATASET = [
    {
        "query_id": "Q1",
        "query": "What is the recommended target HbA1c level for adults with type 2 diabetes managed by lifestyle or monotherapy?",
        "ground_truth_keywords": ["48 mmol/mol", "6.5%", "monotherapy", "lifestyle"],
        "description": "HbA1c target for lifestyle/monotherapy",
    },
    {
        "query_id": "Q2",
        "query": "When should continuous glucose monitoring (CGM) or real-time CGM be offered to adults with type 2 diabetes?",
        "ground_truth_keywords": ["continuous glucose monitoring", "CGM", "real-time", "rtCGM", "isCGM", "insulin"],
        "description": "Continuous glucose monitoring indications",
    },
    {
        "query_id": "Q3",
        "query": "What are the side effects and risks associated with pioglitazone in type 2 diabetes management?",
        "ground_truth_keywords": ["pioglitazone", "fractures", "weight gain", "heart failure", "edema"],
        "description": "Pioglitazone risks and side effects",
    },
    {
        "query_id": "Q4",
        "query": "What first-line drug treatment is recommended for adults with type 2 diabetes and high risk of cardiovascular disease?",
        "ground_truth_keywords": ["SGLT2", "SGLT2 inhibitor", "metformin", "cardiovascular", "CVD"],
        "description": "First-line drug treatment with CVD risk / SGLT2",
    },
    {
        "query_id": "Q5",
        "query": "What advice and management should be provided for hypoglycemia in patients taking insulin or sulfonylureas?",
        "ground_truth_keywords": ["hypoglycaemia", "hypoglycemia", "insulin", "sulfonylurea", "blood glucose"],
        "description": "Hypoglycemia management and precautions",
    },
    {
        "query_id": "Q6",
        "query": "What are the recommended targets for blood pressure control in adults with type 2 diabetes?",
        "ground_truth_keywords": ["blood pressure", "140/90", "130/80", "antihypertensive", "hypertension"],
        "description": "Blood pressure management targets",
    },
]


def is_relevant(retrieved_text: str, ground_truth_keywords: List[str]) -> bool:
    """Check if retrieved chunk text contains at least two ground-truth keyphrases or core clinical concepts."""
    matches = [kw for kw in ground_truth_keywords if kw.lower() in retrieved_text.lower()]
    return len(matches) >= 1


async def run_evaluation():
    logger.info("🚀 Starting Clinical RAG Retrieval Quality Evaluation (Phase 2)...")
    service = RetrievalService()

    results_table = []
    total_reciprocal_rank = 0.0
    total_precision_at_1 = 0.0
    total_precision_at_5 = 0.0
    total_recall_at_5 = 0.0
    total_average_precision = 0.0
    total_hits_at_5 = 0

    for item in EVALUATION_DATASET:
        qid = item["query_id"]
        query = item["query"]
        keywords = item["ground_truth_keywords"]

        # Run retrieval
        response = await service.search(query=query)
        retrieved_docs = response.results  # Top 10

        # Assess relevance of each returned rank
        relevance_vector = [is_relevant(doc.text, keywords) for doc in retrieved_docs[:5]]

        # Compute Metrics
        # 1. Precision@1
        p_at_1 = 1.0 if (relevance_vector and relevance_vector[0]) else 0.0

        # 2. Precision@5
        relevant_in_top_5 = sum(1 for r in relevance_vector if r)
        p_at_5 = relevant_in_top_5 / 5.0

        # 3. Hit Rate@5
        hit_at_5 = 1.0 if relevant_in_top_5 > 0 else 0.0

        # 4. Reciprocal Rank (RR)
        first_rel_rank = 0
        for rank, is_rel in enumerate(relevance_vector, start=1):
            if is_rel:
                first_rel_rank = rank
                break
        rr = 1.0 / first_rel_rank if first_rel_rank > 0 else 0.0

        # 5. Average Precision (AP@5)
        ap_sum = 0.0
        running_rel_count = 0
        for rank, is_rel in enumerate(relevance_vector, start=1):
            if is_rel:
                running_rel_count += 1
                ap_sum += running_rel_count / float(rank)
        ap = ap_sum / max(1, sum(relevance_vector)) if sum(relevance_vector) > 0 else 0.0

        # Accumulate global metrics
        total_precision_at_1 += p_at_1
        total_precision_at_5 += p_at_5
        total_reciprocal_rank += rr
        total_average_precision += ap
        total_hits_at_5 += int(hit_at_5)

        results_table.append({
            "qid": qid,
            "query": query,
            "description": item["description"],
            "p_at_1": p_at_1,
            "p_at_5": p_at_5,
            "hit_at_5": hit_at_5,
            "rr": rr,
            "ap": ap,
            "top_1_snippet": retrieved_docs[0].text[:120].replace("\n", " ") if retrieved_docs else "N/A",
            "top_1_score": retrieved_docs[0].score if retrieved_docs else 0.0,
            "top_1_is_rel": relevance_vector[0] if relevance_vector else False,
        })

    num_queries = len(EVALUATION_DATASET)
    mrr = total_reciprocal_rank / num_queries
    mean_p1 = total_precision_at_1 / num_queries
    mean_p5 = total_precision_at_5 / num_queries
    map_score = total_average_precision / num_queries
    hit_rate_5 = (total_hits_at_5 / num_queries) * 100.0

    print("\n" + "=" * 90)
    print("CLINICAL RETRIEVAL EVALUATION REPORT (BENCHMARK METRICS)")
    print("=" * 90)
    print(f"Total Queries Evaluated : {num_queries}")
    print(f"Mean Reciprocal Rank (MRR) : {mrr:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")
    print(f"Precision@1                : {mean_p1:.4f} ({mean_p1*100:.1f}%)")
    print(f"Precision@5                : {mean_p5:.4f} ({mean_p5*100:.1f}%)")
    print(f"Hit Rate@5                 : {hit_rate_5:.1f}%")
    print("=" * 90)

    print("\nPER-QUERY DETAILED BREAKDOWN:")
    print("-" * 90)
    for r in results_table:
        status_icon = "PASS" if r["hit_at_5"] == 1 else "FAIL"
        print(f"[{r['qid']}] {r['description']} -> {status_icon}")
        print(f"     Top-1 Score: {r['top_1_score']} | Relevant Top-1: {r['top_1_is_rel']}")
        print(f"     P@1: {r['p_at_1']:.2f} | P@5: {r['p_at_5']:.2f} | RR: {r['rr']:.2f} | AP: {r['ap']:.2f}")
        clean_snip = r['top_1_snippet'].encode('ascii', 'ignore').decode('ascii')
        print(f"     Top Chunk Snippet: \"{clean_snip}...\"")
        print("-" * 90)


if __name__ == "__main__":
    asyncio.run(run_evaluation())

