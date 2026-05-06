"""Gold evaluation corpus with 4 papers and 24 curated research questions."""

GOLD_PAPERS = [
    {
        "id": "paper-1",
        "title": "Attention Is All You Need",
        "filename": "vaswani2017_attention.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and entirely relying on an attention mechanism to draw global dependencies between input and output.",
                "bbox": [50.0, 100.0, 500.0, 150.0],
            },
            {
                "page_number": 2,
                "text": "An attention function can be described as mapping a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.",
                "bbox": [50.0, 200.0, 500.0, 250.0],
            },
            {
                "page_number": 3,
                "text": "Scaled Dot-Product Attention: We compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed into matrices K and V. We compute the matrix of outputs as softmax(QK^T / sqrt(d_k))V.",
                "bbox": [50.0, 300.0, 500.0, 350.0],
            },
            {
                "page_number": 4,
                "text": "Multi-Head Attention: Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections.",
                "bbox": [50.0, 400.0, 500.0, 450.0],
            },
            {
                "page_number": 5,
                "text": "Positional Encoding: Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens.",
                "bbox": [50.0, 500.0, 500.0, 550.0],
            },
            {
                "page_number": 6,
                "text": "On the WMT 2014 English-to-German translation task, the big transformer model achieves a state-of-the-art BLEU score of 28.4, outperforming the best existing models including ensembles.",
                "bbox": [50.0, 600.0, 500.0, 650.0],
            },
        ],
    },
    {
        "id": "paper-2",
        "title": "Quantum Supremacy Using a Programmable Superconducting Processor",
        "filename": "arute2019_quantum_supremacy.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical computer. We demonstrate quantum supremacy using a programmable superconducting processor named Sycamore.",
                "bbox": [60.0, 100.0, 510.0, 150.0],
            },
            {
                "page_number": 2,
                "text": "Our Sycamore processor consists of a two-dimensional array of 54 transmon qubits, where each qubit is capacitively coupled to four nearest neighbors in a square lattice geometry.",
                "bbox": [60.0, 200.0, 510.0, 250.0],
            },
            {
                "page_number": 3,
                "text": "Cross-entropy benchmarking fidelity: We assess the fidelity of our quantum circuits by comparing the probability distribution of measured bitstrings with the simulated classical probabilities.",
                "bbox": [60.0, 300.0, 510.0, 350.0],
            },
            {
                "page_number": 4,
                "text": "The Sycamore processor took approximately 200 seconds to sample one instance of a quantum circuit a million times, whereas a state-of-the-art classical supercomputer would require 10,000 years for the same task.",
                "bbox": [60.0, 400.0, 510.0, 450.0],
            },
            {
                "page_number": 5,
                "text": "Quantum error correction will be required to achieve fault-tolerant general-purpose quantum computing. Surface codes with physical error rates below threshold are demonstrated.",
                "bbox": [60.0, 500.0, 510.0, 550.0],
            },
            {
                "page_number": 6,
                "text": "Calibration of single-qubit microwave drive pulses and two-qubit couplers is performed using randomized benchmarking and continuous parameter optimization.",
                "bbox": [60.0, 600.0, 510.0, 650.0],
            },
        ],
    },
    {
        "id": "paper-3",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "filename": "devlin2018_bert.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent models, BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
                "bbox": [55.0, 100.0, 505.0, 150.0],
            },
            {
                "page_number": 2,
                "text": "The Masked Language Model (MLM): In order to train a deep bidirectional representation, we randomly mask 15% of the input tokens at random, and then predict those masked tokens.",
                "bbox": [55.0, 200.0, 505.0, 250.0],
            },
            {
                "page_number": 3,
                "text": "Next Sentence Prediction (NSP): Many important downstream tasks such as Question Answering and Natural Language Inference are based on understanding the relationship between two sentences. We pre-train for a binarized next sentence prediction task.",
                "bbox": [55.0, 300.0, 505.0, 350.0],
            },
            {
                "page_number": 4,
                "text": "BERT architecture is a multi-layer bidirectional Transformer encoder based on the original Transformer implementation described in Vaswani et al.",
                "bbox": [55.0, 400.0, 505.0, 450.0],
            },
            {
                "page_number": 5,
                "text": "On the GLUE benchmark, BERT achieves an overall score of 80.5%, obtaining substantial improvements across all nine individual tasks including MNLI, QQP, and SST-2.",
                "bbox": [55.0, 500.0, 505.0, 550.0],
            },
            {
                "page_number": 6,
                "text": "Fine-tuning BERT on SQuAD v1.1 question answering achieves 93.2 F1 score, outperforming human performance on the benchmark.",
                "bbox": [55.0, 600.0, 505.0, 650.0],
            },
        ],
    },
    {
        "id": "paper-4",
        "title": "CRANE: Citation-grounded Retrieval-Augmented Generation for Scientific Research",
        "filename": "crane2024_grounding.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Retrieval-augmented generation often suffers from hallucinations and misattributed citations where generated text points to incorrect document pages or fabricated statements.",
                "bbox": [45.0, 100.0, 495.0, 150.0],
            },
            {
                "page_number": 2,
                "text": "Verbatim text span alignment maps retrieved chunk elements directly to PDF selectable text layer character spans, eliminating arbitrary percentage overlays and fake bounding boxes.",
                "bbox": [45.0, 200.0, 495.0, 250.0],
            },
            {
                "page_number": 3,
                "text": "Hybrid retrieval combines dense vector embeddings with PostgreSQL tsvector full-text search, merged through reciprocal rank fusion with parameter k=60.",
                "bbox": [45.0, 300.0, 495.0, 350.0],
            },
            {
                "page_number": 4,
                "text": "Multilingual cross-encoder reranking re-evaluates top 40 candidate chunks to distill the top 6 evidence items with minimal positional bias.",
                "bbox": [45.0, 400.0, 495.0, 450.0],
            },
            {
                "page_number": 5,
                "text": "Bounding box origin normalization converts PDF bottom-left points into standard top-left DOM coordinates for synchronized PDF.js viewer rendering.",
                "bbox": [45.0, 500.0, 495.0, 550.0],
            },
            {
                "page_number": 6,
                "text": "Claim-to-citation validation checks semantic support between the generated assertion and the referenced source span, demoting unsupported claims to unresolved status.",
                "bbox": [45.0, 600.0, 495.0, 650.0],
            },
        ],
    },
]

GOLD_QUESTIONS = [
    # Paper 1 Questions (Attention)
    {
        "id": "Q1",
        "query": "How does the Transformer model replace recurrence in sequence transduction?",
        "target_paper_idx": 0,
        "target_page": 1,
        "key_phrase": "entirely relying on an attention mechanism",
    },
    {
        "id": "Q2",
        "query": "What is the mathematical formulation of Scaled Dot-Product Attention?",
        "target_paper_idx": 0,
        "target_page": 3,
        "key_phrase": "softmax(QK^T / sqrt(d_k))V",
    },
    {
        "id": "Q3",
        "query": "Why is multi-head attention beneficial compared to single attention?",
        "target_paper_idx": 0,
        "target_page": 4,
        "key_phrase": "linearly project the queries, keys and values h times",
    },
    {
        "id": "Q4",
        "query": "Why does the Transformer require positional encodings?",
        "target_paper_idx": 0,
        "target_page": 5,
        "key_phrase": "contains no recurrence and no convolution",
    },
    {
        "id": "Q5",
        "query": "What BLEU score did Transformer achieve on WMT 2014 English-to-German?",
        "target_paper_idx": 0,
        "target_page": 6,
        "key_phrase": "state-of-the-art BLEU score of 28.4",
    },
    {
        "id": "Q6",
        "query": "How is the output of an attention function computed from queries and values?",
        "target_paper_idx": 0,
        "target_page": 2,
        "key_phrase": "computed as a weighted sum of the values",
    },
    # Paper 2 Questions (Quantum)
    {
        "id": "Q7",
        "query": "What was the name and architecture of Google's superconducting quantum processor?",
        "target_paper_idx": 1,
        "target_page": 2,
        "key_phrase": "array of 54 transmon qubits",
    },
    {
        "id": "Q8",
        "query": "How long did Sycamore take to sample quantum circuits compared to supercomputers?",
        "target_paper_idx": 1,
        "target_page": 4,
        "key_phrase": "approximately 200 seconds to sample one instance",
    },
    {
        "id": "Q9",
        "query": "How is fidelity measured in cross-entropy benchmarking?",
        "target_paper_idx": 1,
        "target_page": 3,
        "key_phrase": "comparing the probability distribution of measured bitstrings",
    },
    {
        "id": "Q10",
        "query": "What quantum error correction codes are demonstrated?",
        "target_paper_idx": 1,
        "target_page": 5,
        "key_phrase": "Surface codes with physical error rates below threshold",
    },
    {
        "id": "Q11",
        "query": "How are single-qubit microwave drive pulses calibrated?",
        "target_paper_idx": 1,
        "target_page": 6,
        "key_phrase": "randomized benchmarking and continuous parameter optimization",
    },
    {
        "id": "Q12",
        "query": "What is the computational task demonstrating quantum supremacy?",
        "target_paper_idx": 1,
        "target_page": 1,
        "key_phrase": "executed exponentially faster on a quantum processor",
    },
    # Paper 3 Questions (BERT)
    {
        "id": "Q13",
        "query": "What does the BERT acronym stand for?",
        "target_paper_idx": 2,
        "target_page": 1,
        "key_phrase": "Bidirectional Encoder Representations from Transformers",
    },
    {
        "id": "Q14",
        "query": "What percentage of tokens are masked in the Masked Language Model?",
        "target_paper_idx": 2,
        "target_page": 2,
        "key_phrase": "randomly mask 15% of the input tokens",
    },
    {
        "id": "Q15",
        "query": "Why is the Next Sentence Prediction task useful for NLP models?",
        "target_paper_idx": 2,
        "target_page": 3,
        "key_phrase": "understanding the relationship between two sentences",
    },
    {
        "id": "Q16",
        "query": "What score did BERT obtain on the GLUE benchmark?",
        "target_paper_idx": 2,
        "target_page": 5,
        "key_phrase": "overall score of 80.5%",
    },
    {
        "id": "Q17",
        "query": "What performance did fine-tuned BERT achieve on SQuAD v1.1?",
        "target_paper_idx": 2,
        "target_page": 6,
        "key_phrase": "93.2 F1 score, outperforming human performance",
    },
    {
        "id": "Q18",
        "query": "What is the foundational encoder architecture of BERT?",
        "target_paper_idx": 2,
        "target_page": 4,
        "key_phrase": "multi-layer bidirectional Transformer encoder",
    },
    # Paper 4 Questions (CRANE)
    {
        "id": "Q19",
        "query": "What problems affect conventional retrieval-augmented generation?",
        "target_paper_idx": 3,
        "target_page": 1,
        "key_phrase": "hallucinations and misattributed citations",
    },
    {
        "id": "Q20",
        "query": "How does verbatim text span alignment improve citation highlighting?",
        "target_paper_idx": 3,
        "target_page": 2,
        "key_phrase": "eliminating arbitrary percentage overlays and fake bounding boxes",
    },
    {
        "id": "Q21",
        "query": "How are dense vectors and PostgreSQL tsvector queries combined?",
        "target_paper_idx": 3,
        "target_page": 3,
        "key_phrase": "reciprocal rank fusion with parameter k=60",
    },
    {
        "id": "Q22",
        "query": "What is the role of cross-encoder reranking on candidate chunks?",
        "target_paper_idx": 3,
        "target_page": 4,
        "key_phrase": "re-evaluates top 40 candidate chunks to distill the top 6",
    },
    {
        "id": "Q23",
        "query": "How does coordinate normalization handle PDF bottom-left coordinates?",
        "target_paper_idx": 3,
        "target_page": 5,
        "key_phrase": "converts PDF bottom-left points into standard top-left DOM coordinates",
    },
    {
        "id": "Q24",
        "query": "How are unsupported claims handled during citation validation?",
        "target_paper_idx": 3,
        "target_page": 6,
        "key_phrase": "demoting unsupported claims to unresolved status",
    },
]
