# End-to-End Pipeline & Protection Benchmark (Sample Size: 15)

## 1. Grounding & Answer Quality

| Metric | Score |
|:---|:---:|
| **Exact Match (EM)** | 0.0% |
| **Token F1** | 1.5% |
| **Numeric Match** | 0.0% |
| **Average Latency** | 1578.2 ms |

## 2. Claim Factuality & Verification

| Claim Metric | Value |
|:---|:---:|
| Total Claims Extracted | 16 |
| Supported Claims | 3 (18.8%) |
| Contradicted Claims | 0 (0.0%) |
| Unsupported Claims | 13 (81.2%) |
| Mean Factual Consistency | 13.3% |

## 3. Hallucination Protection & Abstention

| Protection Metric | Value |
|:---|:---:|
| Total Queries Evaluated | 15 |
| Accepted Answers | 2 |
| Abstained Answers | 13 |
| **Abstention Rate** | 86.7% |

### Abstention Reasons Breakdown

| Reason | Count |
|:---|:---:|
| `INSUFFICIENT_EVIDENCE` | 13 |
