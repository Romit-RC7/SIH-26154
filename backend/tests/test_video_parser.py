"""Unit tests for VideoParser frame-diff calculation and keyframe sampling."""

import pytest
from PIL import Image, ImageDraw
from backend.app.processors.video_parser import VideoParser


def test_compute_frame_diff_identical():
    img1 = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img2 = Image.new("RGB", (640, 480), color=(255, 0, 0))
    diff = VideoParser._compute_frame_diff(img1, img2)
    assert diff == 0.0


def test_compute_frame_diff_distinct():
    img1 = Image.new("RGB", (640, 480), color=(255, 255, 255))
    img2 = Image.new("RGB", (640, 480), color=(0, 0, 0))
    diff = VideoParser._compute_frame_diff(img1, img2)
    assert diff == 1.0


def test_compute_frame_diff_partial_change():
    img1 = Image.new("RGB", (640, 480), color=(255, 255, 255))
    img2 = Image.new("RGB", (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(img2)
    draw.rectangle([0, 0, 320, 480], fill=(0, 0, 0))
    diff = VideoParser._compute_frame_diff(img1, img2)
    assert 0.4 <= diff <= 0.6


def test_sample_candidate_timestamps():
    timestamps = VideoParser._sample_candidate_timestamps(10.0)
    assert len(timestamps) > 0
    assert timestamps[0] == 0.0
