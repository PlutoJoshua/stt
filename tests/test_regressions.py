import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

import app as web_app
import cli
import processor
from stt_strategies import BaseSTTStrategy, DiarizeWhisperStrategy
from summarizer_strategies import build_summary_instruction


class TranscriptFormattingTests(unittest.TestCase):
    def setUp(self):
        self.strategy = BaseSTTStrategy()

    def test_formats_openai_sdk_response_objects(self):
        response = SimpleNamespace(
            text="전체 텍스트",
            segments=[SimpleNamespace(start=1.6, text=" 안녕하세요 ")],
        )

        self.assertEqual(
            self.strategy._format_transcript_with_timestamps(response),
            "[0:00:02] 안녕하세요",
        )

    def test_formats_dictionary_responses(self):
        response = {"segments": [{"start": 0, "text": " 테스트 "}]}

        self.assertEqual(
            self.strategy._format_transcript_with_timestamps(response),
            "[0:00:00] 테스트",
        )


class DiarizationTests(unittest.TestCase):
    def test_combines_word_start_and_end_fields(self):
        strategy = object.__new__(DiarizeWhisperStrategy)
        turn = SimpleNamespace(start=0, end=2)
        diarization = SimpleNamespace(
            itertracks=lambda yield_label: [(turn, None, "SPEAKER_00")]
        )
        whisper_result = {
            "segments": [
                {"words": [{"word": "안녕하세요", "start": 0.1, "end": 0.8}]}
            ]
        }

        result = strategy._combine_results(diarization, whisper_result)

        self.assertIn("SPEAKER_00", result)
        self.assertIn("안녕하세요", result)


class CliTests(unittest.TestCase):
    def test_cli_passes_a_list_of_audio_files(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("sample.wav").touch()
            with patch("cli.process_file", return_value={}) as process_file:
                result = runner.invoke(
                    cli.process_audio_command,
                    ["sample.wav", "--no-summary"],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(process_file.call_args.kwargs["audio_files"], ["sample.wav"])


class SummaryPromptTests(unittest.TestCase):
    def test_every_ui_summary_type_has_a_real_prompt(self):
        for summary_type in (
            "general",
            "meeting",
            "lecture",
            "interview",
            "daily_conversation",
        ):
            prompt = build_summary_instruction(
                summary_type,
                include_timestamps=True,
                is_bullet_points=True,
            )
            self.assertNotIn("...", prompt)
            self.assertIn("한국어", prompt)
            self.assertIn("타임스탬프", prompt)


class WebSafetyAndEventsTests(unittest.TestCase):
    def test_markdown_does_not_render_raw_html(self):
        rendered = web_app.md.render('<img src=x onerror="alert(1)">')

        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;img", rendered)

    def test_completed_events_are_replayed_to_late_subscribers(self):
        class FakeRedis:
            def exists(self, key):
                return True

            def xread(self, streams, count, block):
                return [
                    (
                        "job_events:test-job",
                        [
                            ("1-0", {"data": json.dumps({"message": "처리 중"})}),
                            ("2-0", {"data": json.dumps({"stage": "complete"})}),
                            ("3-0", {"data": "__STREAM_END__"}),
                        ],
                    )
                ]

        with patch.object(web_app, "redis_client", FakeRedis()):
            response = web_app.app.test_client().get("/status/test-job")
            body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-cache")
        self.assertEqual(response.headers["X-Accel-Buffering"], "no")
        self.assertIn('"stage": "complete"', body)
        self.assertIn("id: 2-0", body)

    def test_redis_read_timeout_keeps_sse_connection_alive(self):
        class FakeRedis:
            def __init__(self):
                self.calls = 0

            def exists(self, key):
                return True

            def xread(self, streams, count, block):
                self.calls += 1
                if self.calls == 1:
                    raise web_app.redis.exceptions.TimeoutError("timed out")
                return [
                    (
                        "job_events:test-job",
                        [
                            ("1-0", {"data": json.dumps({"stage": "complete"})}),
                            ("2-0", {"data": "__STREAM_END__"}),
                        ],
                    )
                ]

        with patch.object(web_app, "redis_client", FakeRedis()):
            response = web_app.app.test_client().get("/status/test-job")
            body = response.get_data(as_text=True)

        self.assertIn(": keep-alive", body)
        self.assertIn('"stage": "complete"', body)

    def test_failed_job_emits_error_without_false_completion_and_cleans_upload(self):
        class FakeRedis:
            def __init__(self):
                self.hashes = {"job:test-job": {"start_time": "0"}}
                self.events = []
                self.expirations = {}

            def hset(self, key, field=None, value=None, mapping=None):
                values = mapping if mapping is not None else {field: value}
                self.hashes.setdefault(key, {}).update(values)

            def hget(self, key, field):
                return self.hashes.get(key, {}).get(field)

            def expire(self, key, seconds):
                self.expirations[key] = seconds

            def xadd(self, key, fields, maxlen, approximate):
                self.events.append(fields["data"])
                return f"{len(self.events)}-0"

        fake_redis = FakeRedis()
        with tempfile.TemporaryDirectory() as temp_dir:
            upload = Path(temp_dir) / "upload.wav"
            upload.write_bytes(b"audio")
            with (
                patch.object(web_app, "redis_client", fake_redis),
                patch.object(web_app, "process_file", side_effect=RuntimeError("실패")),
                patch.object(web_app.time, "time", return_value=1),
            ):
                web_app.run_background_processing("test-job", [str(upload)], {})

            self.assertFalse(upload.exists())

        event_payloads = [json.loads(event) for event in fake_redis.events[:-1]]
        self.assertEqual(fake_redis.hashes["job:test-job"]["status"], "error")
        self.assertTrue(any(event.get("stage") == "error" for event in event_payloads))
        self.assertFalse(any(event.get("stage") == "complete" for event in event_payloads))
        self.assertEqual(fake_redis.events[-1], "__STREAM_END__")
        self.assertEqual(
            fake_redis.expirations["job:test-job"],
            web_app.JOB_TTL_SECONDS,
        )


class ProcessorTests(unittest.TestCase):
    class FakeAudioProcessor:
        converted_paths = []
        stt_methods = []

        def get_audio_info(self, audio_file):
            return {"duration": 1, "duration_formatted": "0:01", "file_size_mb": 1}

        def convert_to_wav(self, audio_file, stt_method=None):
            converted = Path(audio_file).with_name(f"{Path(audio_file).stem}_converted.wav")
            converted.write_bytes(b"wav")
            self.converted_paths.append(converted)
            self.stt_methods.append(stt_method)
            return str(converted)

    class FakeSTTService:
        method = "WhisperAPIStrategy"

        def transcribe(self, audio_file):
            return "변환된 텍스트"

    def setUp(self):
        self.FakeAudioProcessor.converted_paths = []
        self.FakeAudioProcessor.stt_methods = []

    def _run(self, audio_file, output_dir):
        return processor.process_file(
            audio_files=[str(audio_file)],
            output_dir=str(output_dir),
            stt_method="whisper_api",
            summarize_method=None,
            summary_type="general",
            context_file=None,
            no_summary=True,
            bullet_points=False,
            include_timestamps_in_summary=False,
        )

    def test_output_names_do_not_collide_and_temp_files_are_removed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_file = root / "sample.mp3"
            audio_file.write_bytes(b"audio")

            with (
                patch("processor.AudioProcessor", self.FakeAudioProcessor),
                patch("processor.get_stt_service", return_value=self.FakeSTTService()),
            ):
                first = self._run(audio_file, root)
                second = self._run(audio_file, root)

            self.assertNotEqual(first["transcript_file"], second["transcript_file"])
            self.assertTrue(Path(first["transcript_file"]).exists())
            self.assertTrue(Path(second["transcript_file"]).exists())
            self.assertEqual(self.FakeAudioProcessor.stt_methods, ["whisper_api"] * 2)
            self.assertTrue(
                all(not path.exists() for path in self.FakeAudioProcessor.converted_paths)
            )


if __name__ == "__main__":
    unittest.main()
