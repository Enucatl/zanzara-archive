"""CPU checks for timestamped TXT, SRT and WebVTT transcript views."""

from zanzara_archive.contracts import TimedWord
from zanzara_archive.transcripts import render_exports, render_srt, render_txt, render_vtt

WORDS = (
    TimedWord("one", "ciao", 1_250, 2_500, speaker_id="SPEAKER_A"),
    TimedWord("two", "mondo", 3_000, 4_125, speaker_id=None, overlap=True),
)


def test_txt_preserves_offsets_content_labels_and_unassigned_overlap_flags() -> None:
    rendered = render_txt(WORDS)

    assert rendered == (
        "[00:00:01.250 --> 00:00:02.500] [SPEAKER_A] ciao\n"
        "[00:00:03.000 --> 00:00:04.125] [UNASSIGNED] [OVERLAP] mondo\n"
    )


def test_srt_preserves_each_word_as_an_exactly_timed_derived_cue() -> None:
    rendered = render_srt(WORDS)

    assert rendered == (
        "1\n"
        "00:00:01,250 --> 00:00:02,500\n"
        "[SPEAKER_A] ciao\n\n"
        "2\n"
        "00:00:03,000 --> 00:00:04,125\n"
        "[UNASSIGNED] [OVERLAP] mondo\n\n"
    )


def test_vtt_has_header_and_all_cues() -> None:
    rendered = render_vtt(WORDS)

    assert rendered.startswith("WEBVTT\n\n")
    assert "00:00:03.000 --> 00:00:04.125" in rendered
    assert rendered.count("-->") == 2


def test_render_exports_has_only_derived_named_views() -> None:
    rendered = render_exports(WORDS)

    assert set(rendered) == {"transcript.txt", "transcript.srt", "transcript.vtt"}
    assert rendered["transcript.srt"].startswith("1\n")
