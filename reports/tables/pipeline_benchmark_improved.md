# End-to-End Pipeline & Protection Benchmark (Improved Pass) (Sample Size: 15)

## 1. Grounding & Answer Quality

| Metric | Score |
|:---|:---:|
| **Exact Match (EM)** | 6.7% |
| **Token F1** | 2.9% |
| **Numeric Match** | 6.7% |
| **Average Latency** | 1805.7 ms |

## 2. Claim Factuality & Verification

| Claim Metric | Value |
|:---|:---:|
| Total Claims Extracted | 16 |
| Supported Claims | 10 (62.5%) |
| Contradicted Claims | 0 (0.0%) |
| Unsupported Claims | 6 (37.5%) |
| Mean Factual Consistency | 60.0% |

## 3. Hallucination Protection & Abstention

| Protection Metric | Value |
|:---|:---:|
| Total Queries Evaluated | 15 |
| Accepted Answers | 9 |
| Abstained Answers | 6 |
| **Abstention Rate** | 40.0% |

### Abstention Reasons Breakdown

| Reason | Count |
|:---|:---:|
| `INSUFFICIENT_EVIDENCE` | 6 |
