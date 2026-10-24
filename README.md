# LLM, RAG and Vision-Language Models

Hands-on work with modern NLP and multimodal models, from attention mechanics to a working retrieval-augmented generation (RAG) pipeline.

## Highlights
- **RAG pipeline**: document chunking with overlap → sentence-transformer embeddings → ChromaDB vector store → top-k semantic search → answer generation with a seq2seq LLM. **Retrieval Hit@3: 100%, answer accuracy: 89%** on the test questions.
- **Vision-language models**: zero-shot image classification with CLIP, a prompt-engineering study (bare label vs. template vs. domain-specific prompts), and visual question answering with **LLaVA-1.5-7B loaded in 4-bit**.
- **Transformers from first principles**: scaled dot-product attention computed by hand, and image → ViT patch tokens (224×224 → 196 patches of 16×16).
- **Sentiment analysis and NER** with Hugging Face pipelines, plus an LSTM baseline for stock price prediction.

## Notebooks
| Notebook | Topics |
|---|---|
| `01_lstm_sentiment_clip_rag.ipynb` | LSTM forecasting, transformer sentiment analysis, CLIP image–text matching, RAG pipeline |
| `02_attention_vit_tokens_hf_pipelines.ipynb` | Attention by hand, ViT patchification, Hugging Face pipelines (NER, etc.) |
| `03_clip_prompting_and_llava.ipynb` | CLIP zero-shot classification, prompt engineering, LLaVA image understanding |

## Stack
Python · PyTorch · TensorFlow/Keras · Hugging Face Transformers · Sentence-Transformers · ChromaDB · CLIP · LLaVA

Notebooks were run on Google Colab (GPU).
