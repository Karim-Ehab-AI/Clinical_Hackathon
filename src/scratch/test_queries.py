import httpx

client = httpx.Client(timeout=30.0)

queries = [
    {
        "num": 1,
        "title": "هدف نسبة السكر التراكمي (HbA1c Target)",
        "query": "What is the target HbA1c for adults with type 2 diabetes managed by lifestyle or a single drug?",
    },
    {
        "num": 2,
        "title": "دواعي استخدام مراقبة الجلوكوز المستمرة (rtCGM)",
        "query": "When should real-time continuous glucose monitoring rtCGM be offered to adults with type 2 diabetes?",
    },
    {
        "num": 3,
        "title": "مخاطر دواء بايوغليتازون (Pioglitazone Fracture & Weight Risk)",
        "query": "What are the risks of bone fractures and weight gain with pioglitazone?",
    },
    {
        "num": 4,
        "title": "علاج الخط الأول لمريض السكري المعرض لأمراض القلب (SGLT2 & Metformin)",
        "query": "What is the recommended first-line drug treatment for adults with type 2 diabetes at high risk of cardiovascular disease?",
    },
]

for q in queries:
    resp = client.post("http://localhost:3000/api/v1/retrieval/search", json={"query": q["query"]})
    data = resp.json()
    top = data["results"][0]
    clean_text = top['text'][:200].encode('ascii', 'ignore').decode('ascii')
    print(f"=== Q{q['num']} ===")
    print(f"Query: {q['query']}")
    print(f"Expected Top Score: {top['percentage_score']}% ({top['score']})")
    print(f"Expected Snippet: {clean_text}...")
    print(f"Page: {top['pdf_page']} | Document ID: {top['document_id'][:15]}...")
    print("-" * 80)

