"""Uploads at any size: streaming, held-out measurement, no caps, stopping."""

import io
import tracemalloc
import zipfile

import pytest
import torch

from aria import chat as chat_mod
from aria.chat import ChatSession, _command, _dropped_file, _split
from aria.config import LearnerConfig, ModelConfig
from aria.documents import iter_lines
from aria.learner import HOLDOUT_EVERY, HOLDOUT_SEGMENT, OnlineLearner
from aria.model import GPT
from aria.pretrain import create_blank_checkpoint
from aria.tokenizer import BPETokenizer

SENTENCES = [
    "The kettle was on before anyone else woke up.",
    "Rain had been falling on the slate roof all night.",
    "She folded the letter twice and put it in her coat.",
    "Nobody on the street remembered the old bakery.",
    "He walked down to the harbour to watch the boats come in.",
    "The dog slept by the stove until the fire went out.",
]


def numbered_text(n):
    """n distinct sentences, so held-out ones can be recognised exactly."""
    return "\n".join(f"{SENTENCES[i % len(SENTENCES)][:-1]} on day {i}." for i in range(n))


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(" ".join(SENTENCES) * 40 + " on day 0123456789.",
                              vocab_size=400)


@pytest.fixture
def learner(tok, tmp_path):
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=tok.vocab_size, n_layer=2, n_head=4,
                            n_kv_head=2, n_embd=32, block_size=64))
    cfg = LearnerConfig(plasticity="full", learning_rate=1e-3, replay_batch=2,
                        trust_radius=0.0, l2_anchor=0.0, ewc_lambda=0.0,
                        health_check=False, surprise_gate=False, document_batch=1)
    return OnlineLearner(model, tok, cfg, state_dir=tmp_path / "online")


def test_held_out_sentences_are_never_trained_on(learner, tok):
    n_units = HOLDOUT_SEGMENT * HOLDOUT_EVERY * 2
    text = numbered_text(n_units)
    held_days = {i for i in range(n_units)
                 if (i // HOLDOUT_SEGMENT) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1}
    assert held_days

    trained = []
    real_step = learner._step

    def spy(batch, lr):
        for x, _ in batch:
            trained.append(tok.decode(x, skip_special=True))
        return real_step(batch, lr)

    learner._step = spy
    report = learner.learn_document(text, passes=1)
    assert report.heldout, "a document this long should be measured on held-out text"
    seen = " ".join(trained)
    for day in held_days:
        assert f"on day {day}." not in seen, f"held-out sentence {day} was trained on"
    assert "on day 0." in seen


def test_there_is_no_step_cap(learner):
    """Every pass over every page: a long document isn't cut off at some
    fixed number of steps."""
    report = learner.learn_document(numbered_text(700), passes=1)
    assert report.steps > 400
    assert report.steps == report.examples       # document_batch=1


def test_uploads_can_be_stopped_and_keep_their_progress(learner):
    calls = {"n": 0}

    def stop_after_five():
        calls["n"] += 1
        return calls["n"] > 5

    before = {n: p.detach().clone() for n, p in learner.model.named_parameters()}
    report = learner.learn_document(numbered_text(300), passes=3,
                                    should_stop=stop_after_five)
    assert report.stopped and report.steps == 5
    assert any(not torch.equal(before[n], p) for n, p in learner.model.named_parameters())
    assert "stopped early" in report.line()


def test_reading_a_huge_file_uses_flat_memory(tmp_path):
    """The reader streams: a file four times larger costs no more memory."""
    def peak_reading(n_lines):
        path = tmp_path / f"big{n_lines}.txt"
        with open(path, "w") as f:
            for i in range(n_lines):
                f.write(f"{SENTENCES[i % 6]} Line {i}.\n")
        tracemalloc.start()
        n = sum(1 for _ in iter_lines(path, path.name))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert n == n_lines
        return peak

    small, large = peak_reading(40_000), peak_reading(160_000)   # ~2 MB vs ~8 MB
    assert large < small * 1.3, (small, large)
    assert large < 3_000_000


def test_a_line_with_no_newlines_is_split_not_hoarded(tmp_path):
    path = tmp_path / "oneline.txt"
    path.write_text("word " * 200_000)              # 1 MB, no newline at all
    lines = list(iter_lines(path, "oneline.txt"))
    assert len(lines) > 10
    assert max(len(l) for l in lines) <= 16_384


def test_docx_is_streamed_paragraph_by_paragraph(tmp_path):
    paras = "".join(f"<w:p><w:r><w:t>Paragraph {i}.</w:t></w:r></w:p>" for i in range(5000))
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
           f'2006/main"><w:body>{paras}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", xml)
    path = tmp_path / "book.docx"
    path.write_bytes(buf.getvalue())
    lines = list(iter_lines(path, "book.docx"))
    assert lines[0] == "Paragraph 0." and lines[-1] == "Paragraph 4999."


# --- sessions and the terminal ----------------------------------------------


@pytest.fixture
def session(tmp_path):
    ckpt = create_blank_checkpoint(tmp_path / "blank" / "base.pt", size="tiny",
                                   block_size=64)
    return ChatSession(checkpoint=ckpt, max_new_tokens=4)


def test_learning_a_file_from_disk(session, tmp_path):
    doc = tmp_path / "letters.txt"
    doc.write_text(numbered_text(60))
    report, summary = session.learn_file(doc, passes=1)
    assert report.applied and "learned letters.txt" in summary
    assert (session.state_dir / "learned.pt").exists()


def test_long_uploads_save_as_they_go(session, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_mod, "AUTOSAVE_SECONDS", -1)    # save after every step
    saves = []
    real_save = session.save
    monkeypatch.setattr(session, "save", lambda: (saves.append(1), real_save()))
    doc = tmp_path / "doc.txt"
    doc.write_text(numbered_text(40))
    report, _ = session.learn_file(doc, passes=1)
    assert len(saves) >= report.steps


def test_dragged_paths_are_recognised(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("hello")
    assert _dropped_file(str(doc)) == doc
    assert _dropped_file(f"'{doc}'") == doc                   # macOS quotes paths
    assert _dropped_file("notes.txt") is None                 # a bare word is a message
    assert _dropped_file(str(tmp_path / "missing.txt")) is None
    assert _dropped_file(f"{doc} and more") is None
    (tmp_path / "song.mp3").write_bytes(b"x")
    assert _dropped_file(str(tmp_path / "song.mp3")) is None


def test_windows_paths_keep_their_backslashes():
    # Checked on every system: on Windows a backslash separates folders.
    assert _split(r"C:\Users\sam\notes.txt", windows=True) == [r"C:\Users\sam\notes.txt"]
    assert _split(r'"C:\Users\Sam Smith\chat.txt" as Jo', windows=True) == \
        [r"C:\Users\Sam Smith\chat.txt", "as", "Jo"]
    assert _split(r"'D:\my docs\a.txt'", windows=True) == [r"D:\my docs\a.txt"]
    assert _split("'/Users/sam/my notes.txt'", windows=False) == ["/Users/sam/my notes.txt"]


def test_upload_command_asks_whose_voice_to_learn(session, tmp_path, monkeypatch, capsys):
    log = tmp_path / "chat.txt"
    log.write_text("Sam: you coming tonight?\nJo: reckon so, love\n" * 6)
    monkeypatch.setattr("builtins.input", lambda prompt="": "Jo")
    _command(session, f"/upload {log}", interactive=True)
    out = capsys.readouterr().out
    assert "conversation between" in out and "replies by Jo" in out


def test_upload_command_without_a_terminal_does_not_ask(session, tmp_path, capsys):
    log = tmp_path / "chat.txt"
    log.write_text("Sam: you coming tonight?\nJo: reckon so, love\n" * 6)
    _command(session, f"/upload {log}")
    out = capsys.readouterr().out
    assert "learned chat.txt" in out and "name a speaker" in out
