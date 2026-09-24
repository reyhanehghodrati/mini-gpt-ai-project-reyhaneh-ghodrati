"""Regression tests for the completed AI Bootcamp final project."""

import torch
import torch.nn.functional as F

import data_pipeline
import mini_gpt
import tokenizer1
import tokenizer2


def test_tokenizer1_starter_suite() -> None:
    tokenizer1.test_char_tokenizer()
    tokenizer1.test_bpe_tokenizer_get_pairs()
    tokenizer1.test_bpe_tokenizer_merge_pair()
    tokenizer1.test_bpe_tokenizer_train_and_encode_decode()
    tokenizer1.test_bpe_tokenizer_token_to_str()
    tokenizer1.test_compression_ratio()
    tokenizer1.test_vocabulary_stats()


def test_tokenizer2_starter_suite() -> None:
    tokenizer2.test_pre_tokenize()
    tokenizer2.test_apply_merge()
    tokenizer2.test_special_token_handler()
    tokenizer2.test_production_tokenizer_train()
    tokenizer2.test_production_tokenizer_encode_decode()


def test_tokenizer2_multilingual_round_trip() -> None:
    tokenizer = tokenizer2.ProductionTokenizer()
    tokenizer.train("Hello world", num_merges=8)
    tokenizer.add_special_token("<|end|>")
    text = "سلام 你好 🔥 <|end|>"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_data_pipeline_starter_suite() -> None:
    data_pipeline.test_clean_text()
    data_pipeline.test_quality_filter()
    data_pipeline.test_get_shingles()
    data_pipeline.test_minhash_signature()
    data_pipeline.test_lsh_buckets()
    data_pipeline.test_deduplicate()
    data_pipeline.test_simple_tokenizer()
    data_pipeline.test_tokenize_corpus()
    data_pipeline.test_pack_sequences()
    data_pipeline.test_pre_training_data_loader()
    data_pipeline.test_compute_statistics()


def test_minigpt_components_and_gradients() -> None:
    torch.manual_seed(7)
    model = mini_gpt.MiniGPT(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_layers=2,
        max_seq_len=8,
        ff_dim=32,
    )
    token_ids = torch.randint(0, 32, (2, 5))
    targets = torch.randint(0, 32, (2, 5))
    logits = model(token_ids)

    assert logits.shape == (2, 5, 32)
    assert model.count_parameters() == sum(p.numel() for p in model.parameters())

    loss = mini_gpt.cross_entropy_loss(logits, targets)
    reference = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    assert torch.allclose(loss, reference, atol=1e-6)

    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_causal_mask_and_generation_are_reproducible() -> None:
    mask = mini_gpt.causal_mask(5, dtype=torch.float64)
    assert torch.equal(torch.diag(mask), torch.zeros(5, dtype=torch.float64))
    assert mask[0, 1] == torch.finfo(torch.float64).min

    model = mini_gpt.MiniGPT(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        max_seq_len=4,
        ff_dim=32,
    )
    torch.manual_seed(10)
    first = mini_gpt.generate(model, [1, 2], max_new_tokens=5, temperature=0.8)
    torch.manual_seed(10)
    second = mini_gpt.generate(model, [1, 2], max_new_tokens=5, temperature=0.8)
    assert first == second
    assert len(first) == 7


def test_short_training_smoke() -> None:
    torch.manual_seed(42)
    text = "hello transformer world! " * 12
    model = mini_gpt.train_mini_gpt(
        text,
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        seq_len=8,
        num_steps=4,
        lr=1e-3,
        batch_size=2,
    )
    assert model.training is False
