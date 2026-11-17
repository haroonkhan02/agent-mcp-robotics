# Documentation search behind `search_docs`

BM25 over 2,730 sections of the ROS 2 docs, 61 answerable questions from robotics-docs-rag.

| Counted as a hit | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---|---|---|---|---|
| Section from a gold page that contains an answer phrase | 0.410 | 0.607 | 0.672 | 0.787 | 0.526 |
| Any section from a gold page | 0.574 | 0.705 | 0.754 | 0.820 | |

Of the 48 questions with a relevant section in the top 10, 48 have the answer phrase within the first 2,000 characters of that section, which is what one `read_doc_section` call returns.

No relevant section in the top 10: q15, q21, q25, q30, q34, q37, q38, q54, q57, q58, q59, q60, q61.
