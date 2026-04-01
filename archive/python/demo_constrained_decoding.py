#!/usr/bin/env python3
"""
Constrained Decoding 功能演示

本程式示範在吉他譜生成時，如何透過 LogitsProcessor 在每個解碼步驟
只允許符合文法的下一個 token。

用法:
  python demo_constrained_decoding.py
    使用內建範例（兩個音高 64, 67）演示。

  python demo_constrained_decoding.py /path/to/song.gp3
    讀取該 gp3 的 .tokens.txt 及對應 MIDI（若存在），
    並自動對該歌曲的前四個小節做 constrained vs 無 constraint 的演示。

  python demo_constrained_decoding.py --checkpoint outputs/fretting_transformer/best_model.pt ...
    指定模型 checkpoint；會與 training/inference 相同方式載入並用於 constrained / unconstrained 生成。
"""

import argparse
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.tab_dataset import build_vocabulary, collate_fn, events_to_ids
from src.dadagp_parser import parse_dadagp_file, dadagp_to_events
from src.constrained_decoding import (
    TablatureLogitsProcessor,
    build_output_schedule_from_input,
    extract_pitches_from_input_ids,
    STANDARD_TUNING,
)
from src.metrics import compute_tablature_accuracy, generate_and_compute_accuracy
from src.model import FrettingTransformer


def resolve_token_and_midi_paths(gp_path: str, data_dir: Optional[str], midi_dir: Optional[str]):
    """
    由 gp3/gp4/gp5 路徑推得 .tokens.txt 與 MIDI 路徑（與 inference 資料慣例一致）。
    回傳 (token_path, midi_path or None)。
    """
    path = Path(gp_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {gp_path}")

    # Token 檔：DadaGP 慣例為 xxx.gp3.tokens.txt
    if path.suffix.lower() in (".gp3", ".gp4", ".gp5"):
        token_path = path.with_suffix(path.suffix + ".tokens.txt")
    elif path.suffix.lower() == ".txt" and ".tokens" in path.name:
        token_path = path
    else:
        token_path = path.with_name(path.name + ".tokens.txt")
    if not token_path.exists():
        raise FileNotFoundError(f"找不到 token 檔: {token_path}")

    # MIDI：先試同目錄、同主檔名 .mid；若有 data_dir/midi_dir 再試相對路徑
    midi_path = None
    same_dir_midi = path.parent / (path.stem + ".mid")
    if same_dir_midi.exists():
        midi_path = same_dir_midi
    if midi_path is None and data_dir and midi_dir:
        try:
            data_dir = Path(data_dir).resolve()
            midi_dir = Path(midi_dir).resolve()
            path_resolved = path.resolve()
            rel = path_resolved.relative_to(data_dir)
            candidate = midi_dir / rel.with_suffix(".mid")
            if candidate.exists():
                midi_path = candidate
        except ValueError:
            pass
    return str(token_path), str(midi_path) if midi_path else None


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
) -> Tuple[Any, Any, Any, dict]:
    """
    與 inference.py / train 一致：從 checkpoint 讀 config，用 config.data 建 vocab、
    config.model 建 model，再 load state_dict。
    回傳 (model, input_vocab, output_vocab, data_cfg)。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config")
    if not cfg:
        raise ValueError("Checkpoint 缺少 'config'，無法重建 model/vocab。")

    data_cfg = cfg.get("data", {})
    input_vocab, output_vocab = build_vocabulary(
        max_pitch=data_cfg.get("max_pitch", 127),
        max_time_shift=data_cfg.get("max_time_shift", 500),
        num_strings=data_cfg.get("num_strings", 6),
        num_frets=data_cfg.get("num_frets", 25),
        output_format=data_cfg.get("output_format", "v1"),
    )
    model_config = cfg.get("model", {})
    model = FrettingTransformer(
        input_vocab_size=input_vocab.vocab_size,
        output_vocab_size=output_vocab.vocab_size,
        model_config=model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, input_vocab, output_vocab, data_cfg


def load_first_n_bars(
    token_path: str,
    n_bars: int = 4,
    start_bar: int = 1,
    max_pitch: int = 127,
    max_time_shift: int = 500,
    num_strings: int = 6,
    num_frets: int = 25,
    output_format: str = "v1",
    input_vocab=None,
    output_vocab=None,
):
    """
    從 .tokens.txt 載入從 start_bar 開始的 n_bars 個小節 input_ids / output_ids 與詞彙。
    若提供 input_vocab, output_vocab（例如來自 checkpoint），則直接使用，不另建詞彙。
    回傳 (input_ids, output_ids, input_vocab, output_vocab, input_pitches)。
    """
    if input_vocab is None or output_vocab is None:
        input_vocab, output_vocab = build_vocabulary(
            max_pitch=max_pitch,
            max_time_shift=max_time_shift,
            num_strings=num_strings,
            num_frets=num_frets,
            output_format=output_format,
        )
    dadagp_tokens = parse_dadagp_file(token_path)
    input_events, output_events, bar_positions = dadagp_to_events(dadagp_tokens)

    if not bar_positions:
        raise ValueError(f"檔案中沒有任何小節: {token_path}")

    input_ids = events_to_ids(input_events, input_vocab)
    output_ids = events_to_ids(output_events, output_vocab)

    # bar_positions[i] 為第 i+1 小節起始，支援指定起始小節
    total_bars = len(bar_positions)
    if start_bar < 1 or start_bar > total_bars:
        raise ValueError(f"start_bar 超出範圍: {start_bar} (有效範圍: 1..{total_bars})")

    start_idx = start_bar - 1
    end_idx = min(start_idx + max(n_bars, 1), total_bars)

    start_in, start_out = bar_positions[start_idx]
    if end_idx < total_bars:
        end_in, end_out = bar_positions[end_idx][0], bar_positions[end_idx][1]
    else:
        end_in, end_out = len(input_ids), len(output_ids)

    seg_input = input_ids[start_in:end_in]
    seg_output = output_ids[start_out:end_out]
    input_pitches = extract_pitches_from_input_ids(
        torch.tensor(seg_input), input_vocab
    )
    return seg_input, seg_output, input_vocab, output_vocab, input_pitches


def get_valid_token_names(processor, scores_masked):
    """從 masked logits 中取出仍為有效（非 -inf）的 token 名稱。"""
    valid_ids = (scores_masked > float("-inf")).nonzero(as_tuple=True)[0]
    return [processor.vocab.id_to_token.get(i.item(), f"<id={i.item()}>") for i in valid_ids]


def model_generate_unconstrained(
    model: FrettingTransformer,
    input_ids: List[int],
    output_vocab,
    device: str,
    max_length: int = 1024,
    temperature: float = 1.0,
) -> List[int]:
    """與 inference 相同：model.generate 不帶 logits_processor。input_ids 為單一序列，回傳該序列的生成 id list。"""
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(inp, device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inp,
            attention_mask=mask,
            max_length=max_length,
            eos_token_id=output_vocab.eos_id,
            pad_token_id=output_vocab.pad_id,
            temperature=temperature,
        )
    seq = out[0].tolist()
    return seq


def model_generate_constrained(
    model: FrettingTransformer,
    input_ids: List[int],
    output_vocab,
    input_pitches: List[int],
    device: str,
    max_length: int = 1024,
    num_frets: int = 25,
    input_vocab=None,
    strict_input_schedule: bool = False,
) -> List[int]:
    """使用 TablatureLogitsProcessor 做 constrained decoding，並逐步印出每步 constraint 狀態。"""
    schedule = None
    if strict_input_schedule:
        if input_vocab is None:
            raise ValueError("strict_input_schedule requires input_vocab")
        schedule = build_output_schedule_from_input(
            torch.tensor(input_ids, dtype=torch.long),
            input_vocab,
            output_vocab,
        )
    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        num_frets=num_frets,
        device=device,
        strict_input_schedule=strict_input_schedule,
        schedule=schedule,
    )
    processor.reset_state()
    enc_inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(enc_inp, device=device)

    # 與 model.generate 同邏輯，改成單筆逐步解碼，方便每一步印出 constraint。
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_ids=enc_inp,
            attention_mask=attention_mask,
            return_dict=True,
        )
        enc_hidden = encoder_outputs.last_hidden_state

        decoder_input = torch.tensor([[output_vocab.bos_id]], dtype=torch.long, device=device)
        generated = [output_vocab.bos_id]
        past_key_values = None

        print("\n【逐步 Constraint 狀態（模型 constrained decoding）】")
        print(f"  輸入音高序列長度: {len(input_pitches)}")
        print(f"  前 30 個輸入音高: {input_pitches[:30]}{' ...' if len(input_pitches) > 30 else ''}")
        if strict_input_schedule:
            print(f"  strict_input_schedule: True（時程長度 {len(schedule) if schedule else 0}）")
        print()

        for step in range(1, max_length + 1):
            decoder_outputs = model.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_outputs.past_key_values
            logits = model.model.lm_head(decoder_outputs.last_hidden_state[:, -1, :]).squeeze(0)
            masked_logits = processor(torch.tensor(generated, device=device), logits)

            valid_names = get_valid_token_names(processor, masked_logits)
            next_id = int(torch.argmax(masked_logits).item())
            next_token = output_vocab.id_to_token.get(next_id, "?")

            print(f"--- Step {step} ---")
            print(
                "  狀態: "
                f"last_type={processor.last_token_type}, "
                f"active_notes={processor.active_notes[:8]}{' ...' if len(processor.active_notes) > 8 else ''}, "
                f"pitch_idx={processor.pitch_idx}"
                + (
                    f", schedule_idx={processor.schedule_idx}"
                    if strict_input_schedule
                    else ""
                )
            )
            print(f"  本步允許 token 數量: {len(valid_names)}")
            if len(valid_names) <= 25:
                print(f"  允許 token: {valid_names}")
            else:
                print(f"  允許 token (前 12 + 後 5): {valid_names[:12]} ... {valid_names[-5:]}")
            print(f"  -> 模型選擇: {next_token}")
            print()

            generated.append(next_id)
            processor.update_state(next_id)

            if next_id == output_vocab.eos_id:
                break
            decoder_input = torch.tensor([[next_id]], dtype=torch.long, device=device)

    return generated


def build_dummy_encoder_input(input_pitches: List[int], input_vocab, output_format: str = "v1") -> List[int]:
    """無 gp_path 時用內建音高建一筆 encoder 輸入（與 training/inference 的 input 格式一致）。"""
    ids = [input_vocab.bos_id]
    for p in input_pitches:
        note_on = input_vocab.token_to_id.get(f"NOTE_ON_{p}", input_vocab.unk_id)
        ids.append(note_on)
        if output_format != "v3":
            note_off = input_vocab.token_to_id.get(f"NOTE_OFF_{p}", input_vocab.unk_id)
            ids.append(note_off)
    ids.append(input_vocab.token_to_id.get("TIME_SHIFT_240", input_vocab.eos_id))
    ids.append(input_vocab.eos_id)
    return ids


def demo_single_sequence(
    output_vocab=None,
    input_pitches=None,
    title="Constrained Decoding 演示",
    ground_truth_output_ids=None,
    num_frets: int = 25,
    input_vocab=None,
    encoder_input_ids: Optional[List[int]] = None,
    strict_input_schedule: bool = False,
):
    """
    演示單一序列的 constrained decoding 步驟。
    若未提供 output_vocab/input_pitches，則使用內建詞彙與 [64, 67]。
    num_frets 應與 config 一致（例如 25）。
    """
    if output_vocab is None or input_pitches is None:
        _, output_vocab = build_vocabulary(num_frets=num_frets)
        input_pitches = [64, 67]
    device = "cpu"

    schedule = None
    if strict_input_schedule:
        if input_vocab is None or encoder_input_ids is None:
            raise ValueError("strict_input_schedule 需要 input_vocab 與 encoder_input_ids")
        schedule = build_output_schedule_from_input(
            torch.tensor(encoder_input_ids, dtype=torch.long),
            input_vocab,
            output_vocab,
        )

    processor = TablatureLogitsProcessor(
        output_vocab=output_vocab,
        input_pitches=input_pitches,
        num_frets=num_frets,
        device=device,
        strict_input_schedule=strict_input_schedule,
        schedule=schedule,
    )

    seq_ids = [output_vocab.bos_id]
    max_steps = 80 if ground_truth_output_ids else 2048
    step = 0

    print("=" * 70)
    print(title)
    print("=" * 70)
    print("\n【設定】")
    print(f"  輸入音高序列 (來自 encoder): {input_pitches[:30]}{' ...' if len(input_pitches) > 30 else ''}  (共 {len(input_pitches)} 個)")
    print(f"  標準調弦 (open string MIDI): {STANDARD_TUNING}")
    if ground_truth_output_ids:
        gt_tokens = [output_vocab.id_to_token.get(i, "?") for i in ground_truth_output_ids]
        print(f"  前四小節 ground truth 長度: {len(ground_truth_output_ids)} tokens")
    print()

    while step < max_steps:
        step += 1
        input_ids_t = torch.tensor([seq_ids], device=device)
        fake_logits = torch.zeros(output_vocab.vocab_size, device=device)
        masked_logits = processor(input_ids_t[0], fake_logits)
        valid_names = get_valid_token_names(processor, masked_logits)

        seq_tokens = [output_vocab.id_to_token.get(i, "?") for i in seq_ids]
        print(f"--- Step {step} ---")
        print(f"  已生成: {' '.join(seq_tokens[:20])}{' ...' if len(seq_tokens) > 20 else ''}")
        print(f"  狀態: last_type={processor.last_token_type}, active_notes={processor.active_notes[:5]}{' ...' if len(processor.active_notes) > 5 else ''}, pitch_idx={processor.pitch_idx}")
        print(f"  本步允許的 token 數量: {len(valid_names)}")
        if len(valid_names) <= 30:
            print(f"  允許的 token: {valid_names}")
        else:
            print(f"  允許的 token (前 15 + 後 5): {valid_names[:15]} ... {valid_names[-5:]}")

        eos_valid = "EOS" in valid_names
        if eos_valid and step > 6:
            next_id = output_vocab.eos_id
        elif valid_names:
            cand = [t for t in valid_names if t != "EOS"]
            token_str = cand[0] if cand else valid_names[0]
            next_id = output_vocab.token_to_id.get(token_str, output_vocab.eos_id)
        else:
            next_id = output_vocab.eos_id

        seq_ids.append(next_id)
        processor.update_state(next_id)

        if next_id == output_vocab.eos_id:
            print(f"  -> 選擇 EOS，解碼結束。")
            break

        print(f"  -> 選擇: {output_vocab.id_to_token.get(next_id, '?')}")
        print()

    print("\n【最終輸出序列】")
    final_tokens = [output_vocab.id_to_token.get(i, "?") for i in seq_ids]
    print("  " + " ".join(final_tokens[:40]) + (" ..." if len(final_tokens) > 40 else ""))
    print()
    return output_vocab, input_pitches, seq_ids


def demo_unconstrained_sequence(output_vocab, input_pitches, seed=42, max_steps=15):
    """
    使用相同輸入，但不套用 constraint，模擬模型自由生成（此處用隨機取樣模擬可能錯誤）。
    回傳 (seq_ids, seq_tokens)。
    """
    random.seed(seed)
    content_ids = [
        i for i, t in output_vocab.id_to_token.items()
        if t not in ("PAD", "BOS")
    ]
    seq_ids = [output_vocab.bos_id]
    step = 0
    while step < max_steps:
        step += 1
        next_id = random.choice(content_ids)
        seq_ids.append(next_id)
        if next_id == output_vocab.eos_id:
            break
    seq_tokens = [output_vocab.id_to_token.get(i, "?") for i in seq_ids]
    return seq_ids, seq_tokens


def check_violations(seq_tokens, input_pitches):
    """檢查無 constraint 的序列中的文法／語意違規，回傳違規說明列表。"""
    violations = []
    i = 0
    n = len(seq_tokens)
    pitch_idx = 0
    active_notes = []
    while i < n:
        t = seq_tokens[i]
        if t in ("BOS", "EOS"):
            i += 1
            continue
        if t.startswith("NOTE_ON_"):
            try:
                p = int(t.split("_")[2])
            except (IndexError, ValueError):
                i += 1
                continue
            if pitch_idx < len(input_pitches) and p != input_pitches[pitch_idx]:
                violations.append(f"  ・位置 {i}: {t}（輸入期望為 NOTE_ON_{input_pitches[pitch_idx]}）")
            pitch_idx += 1
            active_notes.append(p)
            i += 1
            if i < n and not seq_tokens[i].startswith("TAB_"):
                next_t = seq_tokens[i]
                if next_t.startswith("NOTE_OFF_") or next_t.startswith("TIME_SHIFT_"):
                    violations.append(f"  ・位置 {i}: NOTE_ON 後應接 TAB，卻出現 {next_t}")
            continue
        if t.startswith("TAB_"):
            i += 1
            continue
        if t.startswith("NOTE_OFF_"):
            try:
                p = int(t.split("_")[2])
            except (IndexError, ValueError):
                i += 1
                continue
            if p in active_notes:
                active_notes.remove(p)
            else:
                violations.append(f"  ・位置 {i}: {t} 但當前未開啟此音高（active={active_notes}）")
            i += 1
            continue
        if t.startswith("TIME_SHIFT_"):
            if active_notes:
                violations.append(f"  ・位置 {i}: {t} 時尚有未關閉音符 {active_notes}")
            i += 1
            continue
        i += 1
    if active_notes and "EOS" in seq_tokens:
        violations.append(f"  ・序列以 EOS 結束時仍有未關閉音符: {active_notes}")
    return violations


def print_comparison(
    output_vocab,
    input_pitches,
    constrained_ids,
    unconstrained_tokens,
    ground_truth_tokens=None,
    unconstrained_label=None,
):
    """並列印出 constrained vs 無 constraint（以及 Ground Truth，如有）的輸出與違規說明。"""
    if unconstrained_label is None:
        unconstrained_label = "隨機取樣模擬"
    constrained_tokens = [output_vocab.id_to_token.get(i, "?") for i in constrained_ids]
    print("=" * 70)
    print("對比：有 Constraint vs 無 Constraint（相同輸入）")
    print("=" * 70)
    print(f"\n  輸入音高序列: {input_pitches}")
    print("\n  【有 Constrained Decoding】")
    print("    " + " ".join(constrained_tokens))
    print(f"\n  【無 Constrained Decoding】（{unconstrained_label}）")
    print("    " + " ".join(unconstrained_tokens))
    if ground_truth_tokens is not None:
        print("\n  【Ground Truth（token 檔）】")
        print("    " + " ".join(ground_truth_tokens))
    violations = check_violations(unconstrained_tokens, input_pitches)
    if violations:
        print("\n  無 constraint 序列中的違規：")
        for v in violations:
            print(v)
    else:
        print("\n  （此隨機序列剛好未偵測到明顯違規）")
    print()


def compute_accuracy_vs_ground_truth(
    pred_ids: list,
    target_ids: list,
    output_vocab,
):
    """
    用預測的 token ID 序列與 ground truth 比較，計算 token / pitch / tab accuracy。
    pred_ids 可含 BOS、EOS，會自動去掉並在 EOS 處截斷。
    """
    # 去掉 BOS，並在 EOS 處截斷
    pred_content = list(pred_ids)
    if pred_content and pred_content[0] == output_vocab.bos_id:
        pred_content = pred_content[1:]
    for i, tid in enumerate(pred_content):
        if tid == output_vocab.eos_id:
            pred_content = pred_content[:i]
            break

    pad_id = output_vocab.pad_id
    max_len = max(len(pred_content), len(target_ids))
    pred_padded = pred_content + [pad_id] * (max_len - len(pred_content))
    target_padded = target_ids + [pad_id] * (max_len - len(target_ids))

    pred_t = torch.tensor([pred_padded], dtype=torch.long)
    target_t = torch.tensor([target_padded], dtype=torch.long)
    return compute_tablature_accuracy(
        predictions=pred_t,
        targets=target_t,
        output_vocab=output_vocab,
        # pad_id=pad_id,
    )


def print_token_accuracy(
    output_vocab,
    constrained_ids: list,
    unconstrained_ids: list,
    ground_truth_ids: list,
):
    """印出 Constrained / 無 Constraint 與 ground truth 的 token（及 pitch/tab）accuracy。"""
    metrics_constrained = compute_accuracy_vs_ground_truth(
        constrained_ids, ground_truth_ids, output_vocab
    )
    metrics_unconstrained = compute_accuracy_vs_ground_truth(
        unconstrained_ids, ground_truth_ids, output_vocab
    )
    print("=" * 70)
    print("Token / Pitch / Tab Accuracy（vs 前 N 小節 Ground Truth）")
    print("=" * 70)
    print("\n  【Constrained Decoding】")
    print(f"    Token Accuracy:  {metrics_constrained.token_accuracy:.2%}")
    print(f"    Pitch Accuracy:  {metrics_constrained.pitch_accuracy:.2%}")
    print(f"    Tab Accuracy:    {metrics_constrained.tab_accuracy:.2%}")
    print(f"    Total Tokens:    {metrics_constrained.total_tokens:,}")
    print(f"    Total Notes:     {metrics_constrained.total_notes:,}")
    print("\n  【無 Constrained Decoding】")
    print(f"    Token Accuracy:  {metrics_unconstrained.token_accuracy:.2%}")
    print(f"    Pitch Accuracy:  {metrics_unconstrained.pitch_accuracy:.2%}")
    print(f"    Tab Accuracy:    {metrics_unconstrained.tab_accuracy:.2%}")
    print(f"    Total Tokens:    {metrics_unconstrained.total_tokens:,}")
    print(f"    Total Notes:     {metrics_unconstrained.total_notes:,}")
    print()


class _OneSegmentDataset(Dataset):
    """單筆 (input_ids, output_ids)，與 TabDataset.__getitem__ 回傳格式相同（供 collate_fn 使用）。"""

    def __init__(self, input_ids: List[int], output_ids: List[int]):
        self.input_ids = np.asarray(input_ids, dtype=np.int64)
        self.output_ids = np.asarray(output_ids, dtype=np.int64)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.input_ids, self.output_ids


def make_demo_metrics_dataloader(
    seg_input: List[int],
    seg_output: List[int],
    input_vocab,
    output_vocab,
) -> DataLoader:
    """
    與訓練時 TabDataset 一致：output 序列為 BOS + 內容 + EOS，再經 collate_fn 組 batch。
    """
    out_with_special = [output_vocab.bos_id] + list(seg_output) + [output_vocab.eos_id]
    ds = _OneSegmentDataset(seg_input, out_with_special)
    collate = partial(
        collate_fn,
        input_pad_id=input_vocab.pad_id,
        output_pad_id=output_vocab.pad_id,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)


def print_metrics_via_generate_and_compute_accuracy(
    model: FrettingTransformer,
    seg_input: List[int],
    seg_output: List[int],
    input_vocab,
    output_vocab,
    device: str,
    num_frets: int = 25,
    max_length: int = 1024,
    temperature: float = 1.0,
    strict_input_schedule: bool = False,
):
    """
    與 train.py 相同：呼叫 src.metrics.generate_and_compute_accuracy（含 Difficulty）。
    分別以 constrained / unconstrained 各跑一次單筆 batch。
    """
    loader = make_demo_metrics_dataloader(
        seg_input, seg_output, input_vocab, output_vocab
    )
    print("=" * 70)
    print("AR Metrics（與 train.py 相同：generate_and_compute_accuracy）")
    print("=" * 70)

    ar_constrained, _ = generate_and_compute_accuracy(
        model=model,
        dataloader=loader,
        output_vocab=output_vocab,
        device=device,
        max_length=max_length,
        max_batches=1,
        input_vocab=input_vocab,
        use_teacher_forcing=True,
        temperature=temperature,
        use_constrained_decoding=True,
        num_frets=num_frets,
        strict_input_schedule=strict_input_schedule,
    )
    print("\n  【Constrained Decoding】")
    print(f"    {ar_constrained}")
    print(f"    Difficulty:      {ar_constrained.difficulty:.5f}")

    ar_unconstrained, _ = generate_and_compute_accuracy(
        model=model,
        dataloader=loader,
        output_vocab=output_vocab,
        device=device,
        max_length=max_length,
        max_batches=1,
        input_vocab=input_vocab,
        use_teacher_forcing=True,
        temperature=temperature,
        use_constrained_decoding=False,
        num_frets=num_frets,
    )
    print("\n  【無 Constrained Decoding】")
    print(f"    {ar_unconstrained}")
    print(f"    Difficulty:      {ar_unconstrained.difficulty:.5f}")
    print()


def demo_without_constraint():
    """對比：若無 constrained decoding，模型可能產生的錯誤範例。"""
    print("=" * 70)
    print("對比：無 Constrained Decoding 時常見錯誤")
    print("=" * 70)
    print("""
  若在生成時不套用 constraint，模型可能產生：

  1. 缺少 TAB：NOTE_ON_64 後直接接 NOTE_OFF_64
     → 文法要求 NOTE_ON 後必須接 TAB（指定弦與格數）

  2. 音高不對齊：輸入為 [64, 67]，卻輸出 NOTE_ON_65
     → Constraint 只允許下一個 NOTE_ON 的 pitch 等於 input_pitches[pitch_idx]

  3. TAB 與 NOTE_ON 不符：NOTE_ON_64 後接 TAB_3_5（可能對應其他音高）
     → Constraint 只允許該 pitch 對應的合法 (string, fret) TAB

  4. 未關閉所有音符就 TIME_SHIFT：NOTE_ON_64, TAB_1_0 後直接 TIME_SHIFT_240
     → 必須先 NOTE_OFF_64 才能 TIME_SHIFT

  5. NOTE_OFF 與當前開啟的音不符：只開了 64 卻輸出 NOTE_OFF_67
     → Constraint 只允許關閉目前 active_notes 中的音高

  本專案在 inference 時可透過 config 開啟 constrained decoding：
    training.use_constrained_decoding: true
  以在生成當下就強制符合上述文法，減少事後 post-processing 需求。
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constrained Decoding 演示。可傳入 gp3 路徑以自動讀取該曲前四小節。"
    )
    parser.add_argument(
        "gp_path",
        nargs="?",
        default=None,
        help="選填：.gp3 / .gp4 / .gp5 或 .tokens.txt 路徑；未提供則使用內建範例。",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="選填：與 inference 相同的 data_dir，用於推 MIDI 路徑。",
    )
    parser.add_argument(
        "--midi-dir",
        default=None,
        help="選填：MIDI 目錄，與 data-dir 搭配推對應 .mid 路徑。",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=1,
        help="從起始小節開始，使用幾個小節（預設 1）。",
    )
    parser.add_argument(
        "--target-bar",
        type=int,
        default=None,
        help="只看單一小節（1-based）。若設定則等同 --bars 1 並覆蓋起始小節。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="選填：模型 checkpoint 路徑（.pt）。若提供，之後可接真實模型做 inference；目前 demo 仍以隨機取樣模擬 unconstrained。",
    )
    parser.add_argument(
        "--strict-input-schedule",
        action="store_true",
        help="與 inference 的 strict_input_schedule 相同：NOTE_OFF/TIME_SHIFT 依 encoder 時程鎖定。",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    input_vocab = None
    output_vocab = None
    data_cfg = {}

    if args.checkpoint:
        print("Checkpoint 路徑:", args.checkpoint)
        if not os.path.isfile(args.checkpoint):
            print("  警告: 檔案不存在，請確認路徑。")
        else:
            model, input_vocab, output_vocab, data_cfg = load_model_from_checkpoint(
                args.checkpoint, device
            )

    if args.gp_path:
        start_bar = args.target_bar if args.target_bar is not None else 1
        bars_to_use = 1 if args.target_bar is not None else args.bars
        token_path, midi_path = resolve_token_and_midi_paths(
            args.gp_path, args.data_dir, args.midi_dir
        )
        print("=" * 70)
        if args.target_bar is not None:
            print("從 GP 檔案載入（僅第 {} 小節）".format(start_bar))
        else:
            print("從 GP 檔案載入（第 {} 小節起，共 {} 小節）".format(start_bar, bars_to_use))
        print("=" * 70)
        print(f"  GP/token: {args.gp_path}")
        print(f"  Token 檔: {token_path}")
        print(f"  MIDI 檔: {midi_path or '(未找到，僅用 token 檔)'}")
        print()

        seg_input, seg_output, input_vocab, output_vocab, input_pitches = load_first_n_bars(
            token_path,
            n_bars=bars_to_use,
            start_bar=start_bar,
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            num_frets=data_cfg.get("num_frets", 25),
            output_format=data_cfg.get("output_format", "v1"),
        )
        if args.target_bar is not None:
            print(f"  第 {start_bar} 小節: input tokens {len(seg_input)}, output tokens {len(seg_output)}, 音高數 {len(input_pitches)}\n")
        else:
            print(f"  第 {start_bar} 小節起共 {bars_to_use} 小節: input tokens {len(seg_input)}, output tokens {len(seg_output)}, 音高數 {len(input_pitches)}\n")

        if model is not None:
            print("=" * 70)
            if args.target_bar is not None:
                print("Constrained Decoding 演示（歌曲第 {} 小節，使用模型）".format(start_bar))
            else:
                print("Constrained Decoding 演示（歌曲第 {} 小節起，共 {} 小節，使用模型）".format(start_bar, bars_to_use))
            print("=" * 70)
            constrained_ids = model_generate_constrained(
                model,
                seg_input,
                output_vocab,
                input_pitches,
                device,
                num_frets=data_cfg.get("num_frets", 25),
                input_vocab=input_vocab,
                strict_input_schedule=args.strict_input_schedule,
            )
        else:
            output_vocab, input_pitches, constrained_ids = demo_single_sequence(
                output_vocab=output_vocab,
                input_pitches=input_pitches,
                title=(
                    "Constrained Decoding 演示（歌曲第 {} 小節）".format(start_bar)
                    if args.target_bar is not None
                    else "Constrained Decoding 演示（歌曲第 {} 小節起，共 {} 小節）".format(start_bar, bars_to_use)
                ),
                ground_truth_output_ids=seg_output,
                num_frets=data_cfg.get("num_frets", 25),
            )
            
        gt_tokens = [output_vocab.id_to_token.get(i, "?") for i in seg_output]
        if args.target_bar is not None:
            print("【第 {} 小節 Ground Truth（來自 token 檔）】".format(start_bar))
        else:
            print("【第 {} 小節起共 {} 小節 Ground Truth（來自 token 檔）】".format(start_bar, bars_to_use))
        print("  " + " ".join(gt_tokens[:50]) + (" ..." if len(gt_tokens) > 50 else ""))
        print()
        ground_truth_ids = seg_output
    else:
        if model is not None:
            output_format = data_cfg.get("output_format", "v1")
            encoder_input = build_dummy_encoder_input([64, 67], input_vocab, output_format)
            input_pitches = [64, 67]
            print("=" * 70)
            print("Constrained Decoding 演示（內建範例，使用模型）")
            print("=" * 70)
            constrained_ids = model_generate_constrained(
                model,
                encoder_input,
                output_vocab,
                input_pitches,
                device,
                num_frets=data_cfg.get("num_frets", 25),
                input_vocab=input_vocab,
                strict_input_schedule=args.strict_input_schedule,
            )
        else:
            output_vocab, input_pitches, constrained_ids = demo_single_sequence(
                num_frets=data_cfg.get("num_frets", 25),
            )
        ground_truth_ids = None

    print("=" * 70)
    print("無 Constrained Decoding 重新生成（相同輸入）")
    print("=" * 70)
    if model is not None:
        encoder_input = seg_input if args.gp_path else build_dummy_encoder_input(
            [64, 67], input_vocab, data_cfg.get("output_format", "v1")
        )
        print("\n  使用相同 encoder 輸入，以模型 generate（無 LogitsProcessor）。\n")
        unconstrained_ids = model_generate_unconstrained(
            model, encoder_input, output_vocab, device
        )
        unconstrained_tokens = [output_vocab.id_to_token.get(i, "?") for i in unconstrained_ids]
    else:
        print("\n  使用相同 input_pitches，從整個詞彙隨機取樣模擬模型輸出。\n")
        unconstrained_ids, unconstrained_tokens = demo_unconstrained_sequence(
            output_vocab, input_pitches
        )

    if ground_truth_ids is not None:
        gt_tokens_for_cmp = [
            output_vocab.id_to_token.get(i, "?") for i in ground_truth_ids
        ]
    else:
        gt_tokens_for_cmp = None

    print_comparison(
        output_vocab,
        input_pitches,
        constrained_ids,
        unconstrained_tokens,
        ground_truth_tokens=gt_tokens_for_cmp,
        unconstrained_label="模型生成" if model is not None else "隨機取樣模擬",
    )

    if ground_truth_ids is not None:
        if model is not None:
            print_metrics_via_generate_and_compute_accuracy(
                model=model,
                seg_input=seg_input,
                seg_output=seg_output,
                input_vocab=input_vocab,
                output_vocab=output_vocab,
                device=device,
                num_frets=data_cfg.get("num_frets", 25),
                max_length=1024,
                strict_input_schedule=args.strict_input_schedule,
            )
        else:
            print_token_accuracy(
                output_vocab,
                constrained_ids,
                unconstrained_ids,
                ground_truth_ids,
            )

    # demo_without_constraint()
    print("Done.")
