from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.schedule_batch import PrefillState, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase


def make_req(rid: str, priority: int, wait_time: float, split_index: int = 0):
    flowprefill_ctx = SimpleNamespace(
        split_index=split_index,
        split_forward_batch=None,
        seq_lens_cpu_cache=None,
        split_attn_backend_needs_reinit=False,
        prefill_stats=None,
        resume_batch=None,
    )
    req = SimpleNamespace()
    req.rid = rid
    req.priority = priority
    req.prefill_arrival_ts = wait_time
    req.prefill_deadline_ts = None
    req.prefill_slack = None
    req.prefill_predicted_remaining_time = None
    req.prefill_has_explicit_slack = False
    req.prefill_has_explicit_remaining_time = False
    req.prefill_observed_split_runtime = 0.0
    req.prefill_observed_split_layers = 0
    req.prefill_preempt_pending = False
    req.prefill_num_preemptions = 0
    req.prefill_resume_split_index = split_index
    req.prefill_state = PrefillState.WAITING
    req.to_finish = None
    req.return_logprob = False
    req.grammar = None
    req.input_embeds = None
    req.multimodal_inputs = None
    req.stream = False
    req.return_hidden_states = False
    req.return_routed_experts = False
    req.top_logprobs_num = 0
    req.token_ids_logprob = None
    req.extend_input_len = 1
    req.extend_logprob_start_len = 0
    req.prefix_indices = []
    req.is_prefill_only = False
    req.dimensions = None
    req.finished = lambda: False
    req.time_stats = SimpleNamespace(
        wait_queue_entry_time=wait_time,
        prefill_run_batch_start_time=0.0,
        prefill_run_batch_end_time=0.0,
    )
    req.flowprefill_ctx = flowprefill_ctx

    def sync_flowprefill_ctx_from_batch(batch):
        req.prefill_resume_split_index = batch.split_index
        req.flowprefill_ctx.split_index = batch.split_index
        req.flowprefill_ctx.split_forward_batch = getattr(
            batch, "split_forward_batch", None
        )
        req.flowprefill_ctx.seq_lens_cpu_cache = getattr(
            batch, "seq_lens_cpu_cache", None
        )
        req.flowprefill_ctx.split_attn_backend_needs_reinit = getattr(
            batch, "split_attn_backend_needs_reinit", False
        )
        req.flowprefill_ctx.prefill_stats = getattr(batch, "prefill_stats", None)
        req.flowprefill_ctx.resume_batch = batch

    def apply_flowprefill_ctx_to_batch(batch):
        batch.split_index = req.flowprefill_ctx.split_index
        batch.split_forward_batch = req.flowprefill_ctx.split_forward_batch
        batch.seq_lens_cpu_cache = req.flowprefill_ctx.seq_lens_cpu_cache
        batch.split_attn_backend_needs_reinit = (
            req.flowprefill_ctx.split_attn_backend_needs_reinit
        )
        batch.prefill_stats = req.flowprefill_ctx.prefill_stats

    def reset_flowprefill_ctx():
        req.flowprefill_ctx.split_index = 0
        req.flowprefill_ctx.split_forward_batch = None
        req.flowprefill_ctx.seq_lens_cpu_cache = None
        req.flowprefill_ctx.split_attn_backend_needs_reinit = False
        req.flowprefill_ctx.prefill_stats = None
        req.flowprefill_ctx.resume_batch = None

    req.sync_flowprefill_ctx_from_batch = sync_flowprefill_ctx_from_batch
    req.apply_flowprefill_ctx_to_batch = apply_flowprefill_ctx_to_batch
    req.reset_flowprefill_ctx = reset_flowprefill_ctx
    return req


def make_batch(reqs, split_index: int):
    batch = SimpleNamespace()
    batch.reqs = reqs
    batch.split_index = split_index
    batch.batch_is_full = False
    batch.split_prefill_finished = False
    batch.split_forward_count = 1
    batch.forward_mode = ForwardMode.SPLIT_PREFILL
    batch.split_forward_batch = None
    batch.seq_lens_cpu_cache = None
    batch.split_attn_backend_needs_reinit = False
    batch.prefill_stats = None

    def mark_flowprefill_preempt_pending():
        for req in reqs:
            req.prefill_preempt_pending = True
            req.prefill_state = PrefillState.PREEMPT_PENDING

    batch.mark_flowprefill_preempt_pending = mark_flowprefill_preempt_pending
    return batch


class TestFlowPrefillScheduler(CustomTestCase):
    def setUp(self):
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.server_args = SimpleNamespace(
            default_priority_value=None,
            schedule_low_priority_values_first=False,
            flowprefill_priority_policy="priority_fcfs",
            flowprefill_max_preemptions=0,
            flowprefill_split_layers=1,
            flowprefill_default_ttft_slo_ms=None,
        )
        self.scheduler.schedule_low_priority_values_first = False
        self.scheduler.enable_flowprefill = True
        self.scheduler.policy = MagicMock()
        self.scheduler.policy.calc_priority.side_effect = (
            lambda waiting_queue, running_batch: waiting_queue.sort(
                key=self.scheduler._flowprefill_priority_key
            )
        )
        self.scheduler.current_scheduler_metrics_enabled = False
        self.scheduler.metrics_collector = MagicMock()
        self.scheduler.stats = SimpleNamespace(num_preempted_prefill_queue_reqs=0)
        self.scheduler.waiting_queue = []
        self.scheduler.running_batch = SimpleNamespace(reqs=[])
        self.scheduler.preempted_prefill_queue = deque()
        self.scheduler.running_split_prefill_batch = None
        self.scheduler.flowprefill_observed_split_runtime = 0.0
        self.scheduler.flowprefill_observed_split_layers = 0
        self.scheduler.req_to_token_pool = SimpleNamespace()
        self.scheduler.token_to_kv_pool_allocator = SimpleNamespace()
        self.scheduler.tree_cache = MagicMock()
        self.scheduler.enable_overlap = False
        self.scheduler.spec_algorithm = None
        self.scheduler.model_config = SimpleNamespace(
            num_hidden_layers=4, is_encoder_decoder=False, vocab_size=32000
        )
        self.scheduler.process_batch_result_prefill = MagicMock()

    def test_arrival_marks_running_batch_preempt_pending(self):
        running_req = make_req("running", priority=5, wait_time=10.0)
        running_batch = make_batch([running_req], split_index=1)
        self.scheduler.running_split_prefill_batch = running_batch

        incoming_req = make_req("incoming", priority=10, wait_time=20.0)
        self.scheduler._maybe_mark_flowprefill_preempt_pending(incoming_req)

        self.assertTrue(running_req.prefill_preempt_pending)
        self.assertEqual(running_req.prefill_state, PrefillState.PREEMPT_PENDING)

    def test_deadline_fcfs_prefers_earlier_deadline(self):
        self.scheduler.server_args.flowprefill_priority_policy = "deadline_fcfs"
        req0 = make_req("r0", priority=1, wait_time=10.0)
        req1 = make_req("r1", priority=10, wait_time=5.0)
        req0.prefill_deadline_ts = 200.0
        req1.prefill_deadline_ts = 100.0

        self.assertLess(
            self.scheduler._flowprefill_priority_key(req1),
            self.scheduler._flowprefill_priority_key(req0),
        )

    def test_initialize_flowprefill_deadline_from_ttft_slo(self):
        req = make_req("r0", priority=1, wait_time=10.0)
        req.arrival_time = 10.0
        recv_req = SimpleNamespace(
            prefill_ttft_slo_ms=250.0,
            prefill_deadline_ts=None,
            prefill_slack=None,
            prefill_predicted_remaining_time=None,
        )

        self.scheduler._initialize_flowprefill_scheduling_fields(req, recv_req)

        self.assertEqual(req.prefill_arrival_ts, 10.0)
        self.assertEqual(req.prefill_deadline_ts, 10.25)
        self.assertIsNone(req.prefill_slack)

    def test_initialize_flowprefill_deadline_from_server_default_ttft_slo(self):
        req = make_req("r0", priority=1, wait_time=10.0)
        req.arrival_time = 10.0
        self.scheduler.server_args.flowprefill_default_ttft_slo_ms = 400.0
        recv_req = SimpleNamespace(
            prefill_ttft_slo_ms=None,
            prefill_deadline_ts=None,
            prefill_slack=None,
            prefill_predicted_remaining_time=None,
        )

        self.scheduler._initialize_flowprefill_scheduling_fields(req, recv_req)

        self.assertEqual(req.prefill_deadline_ts, 10.4)

    def test_initialize_flowprefill_preserves_explicit_slack_and_remaining_time(self):
        req = make_req("r0", priority=1, wait_time=10.0)
        req.arrival_time = 10.0
        recv_req = SimpleNamespace(
            prefill_ttft_slo_ms=None,
            prefill_deadline_ts=12.0,
            prefill_slack=1.5,
            prefill_predicted_remaining_time=0.75,
        )

        self.scheduler._initialize_flowprefill_scheduling_fields(req, recv_req)

        self.assertEqual(req.prefill_deadline_ts, 12.0)
        self.assertEqual(req.prefill_slack, 1.5)
        self.assertEqual(req.prefill_predicted_remaining_time, 0.75)

    def test_initialize_flowprefill_predictor_uses_scheduler_average_when_available(self):
        req = make_req("r0", priority=1, wait_time=10.0)
        req.arrival_time = 10.0
        self.scheduler.flowprefill_observed_split_runtime = 0.8
        self.scheduler.flowprefill_observed_split_layers = 4
        recv_req = SimpleNamespace(
            prefill_ttft_slo_ms=600.0,
            prefill_deadline_ts=None,
            prefill_slack=None,
            prefill_predicted_remaining_time=None,
        )

        self.scheduler._initialize_flowprefill_scheduling_fields(req, recv_req)

        self.assertEqual(req.prefill_predicted_remaining_time, 0.8)
        self.assertEqual(req.prefill_slack, -0.2)

    def test_slack_edf_prefers_smaller_slack(self):
        self.scheduler.server_args.flowprefill_priority_policy = "slack_edf"
        req0 = make_req("r0", priority=1, wait_time=10.0)
        req1 = make_req("r1", priority=10, wait_time=5.0)
        req0.prefill_slack = 10.0
        req1.prefill_slack = 1.0
        req0.prefill_deadline_ts = 200.0
        req1.prefill_deadline_ts = 300.0

        self.assertLess(
            self.scheduler._flowprefill_priority_key(req1),
            self.scheduler._flowprefill_priority_key(req0),
        )

    def test_slack_edf_uses_heuristic_remaining_time_when_slack_missing(self):
        self.scheduler.server_args.flowprefill_priority_policy = "slack_edf"
        self.scheduler.flowprefill_observed_split_runtime = 0.4
        self.scheduler.flowprefill_observed_split_layers = 4
        req0 = make_req("r0", priority=1, wait_time=10.0)
        req1 = make_req("r1", priority=10, wait_time=5.0)
        req0.prefill_deadline_ts = 10.8
        req1.prefill_deadline_ts = 10.5
        req0.flowprefill_ctx.split_index = 0
        req1.flowprefill_ctx.split_index = 2

        with patch("sglang.srt.managers.scheduler.time.perf_counter", return_value=10.0):
            self.assertLess(
                self.scheduler._flowprefill_priority_key(req1),
                self.scheduler._flowprefill_priority_key(req0),
            )

    def test_preempted_batch_is_selected_before_lower_priority_waiting(self):
        waiting_req = make_req("waiting", priority=1, wait_time=1.0)
        preempted_req = make_req("preempted", priority=5, wait_time=2.0, split_index=2)
        preempted_batch = make_batch([preempted_req], split_index=2)
        preempted_req.prefill_state = PrefillState.PREEMPTED
        preempted_req.sync_flowprefill_ctx_from_batch(preempted_batch)

        self.scheduler.waiting_queue = [waiting_req]
        self.scheduler.preempted_prefill_queue.append(preempted_req)

        selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, preempted_batch)
        self.assertEqual(selected.split_index, 2)
        self.assertEqual(preempted_req.prefill_state, PrefillState.RUNNING)

    def test_waiting_request_wins_when_higher_priority_than_preempted_batch(self):
        waiting_req = make_req("waiting", priority=10, wait_time=1.0)
        preempted_req = make_req("preempted", priority=5, wait_time=2.0, split_index=3)
        preempted_batch = make_batch([preempted_req], split_index=3)
        preempted_req.sync_flowprefill_ctx_from_batch(preempted_batch)

        self.scheduler.waiting_queue = [waiting_req]
        self.scheduler.preempted_prefill_queue.append(preempted_req)

        selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIsNone(selected)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 1)

    def test_intermediate_split_prefill_is_enqueued_when_preempt_pending(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=1)
        req.prefill_preempt_pending = True
        batch = make_batch([req], split_index=2)
        batch.split_forward_batch = SimpleNamespace(split_index=3)
        req.prefill_deadline_ts = 10.0
        req.time_stats.prefill_run_batch_start_time = 2.0
        req.time_stats.prefill_run_batch_end_time = 2.4
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertIsNone(self.scheduler.running_split_prefill_batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 1)
        self.assertEqual(req.prefill_state, PrefillState.PREEMPTED)
        self.assertEqual(req.prefill_resume_split_index, 3)
        self.assertEqual(req.prefill_observed_split_runtime, 0.4)
        self.assertEqual(req.prefill_observed_split_layers, 1)
        self.assertEqual(req.prefill_predicted_remaining_time, 0.8)

    def test_preempted_queue_stores_requests_not_batches(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=1)
        req1 = make_req("r1", priority=5, wait_time=1.5, split_index=1)
        req0.prefill_preempt_pending = True
        req1.prefill_preempt_pending = True
        batch = make_batch([req0, req1], split_index=2)
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 2)
        self.assertIs(self.scheduler.preempted_prefill_queue[0], req0)
        self.assertIs(self.scheduler.preempted_prefill_queue[1], req1)
        self.assertIs(req0.flowprefill_ctx.resume_batch, batch)
        self.assertIs(req1.flowprefill_ctx.resume_batch, batch)

    def test_multi_req_preempted_batch_prefers_request_owned_state_when_sliceable(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=1)
        req1 = make_req("r1", priority=4, wait_time=1.5, split_index=1)
        req0.prefill_preempt_pending = True
        req1.prefill_preempt_pending = True
        batch = make_batch([req0, req1], split_index=2)
        batch.split_forward_batch = MagicMock()
        batch.split_forward_batch.slice_for_flowprefill_req.side_effect = [
            SimpleNamespace(batch_size=1),
            SimpleNamespace(batch_size=1),
        ]
        batch.seq_lens_cpu_cache = ["seq0", "seq1"]
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 2)
        self.assertIsNone(req0.flowprefill_ctx.resume_batch)
        self.assertIsNone(req1.flowprefill_ctx.resume_batch)
        self.assertIsNotNone(req0.flowprefill_ctx.split_forward_batch)
        self.assertIsNotNone(req1.flowprefill_ctx.split_forward_batch)
        self.assertEqual(req0.flowprefill_ctx.seq_lens_cpu_cache, ["seq0"])
        self.assertEqual(req1.flowprefill_ctx.seq_lens_cpu_cache, ["seq1"])

    def test_sibling_preempted_requests_resume_together(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req1 = make_req("r1", priority=4, wait_time=2.0, split_index=2)
        batch = make_batch([req0, req1], split_index=2)
        req0.prefill_state = PrefillState.PREEMPTED
        req1.prefill_state = PrefillState.PREEMPTED
        req0.sync_flowprefill_ctx_from_batch(batch)
        req1.sync_flowprefill_ctx_from_batch(batch)

        self.scheduler.preempted_prefill_queue.append(req0)
        self.scheduler.preempted_prefill_queue.append(req1)

        selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 0)
        self.assertEqual(req0.prefill_state, PrefillState.RUNNING)
        self.assertEqual(req1.prefill_state, PrefillState.RUNNING)

    def test_sliceable_multi_req_parked_reqs_regroup_without_fallback(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req1 = make_req("r1", priority=4, wait_time=2.0, split_index=2)
        req0.prefill_state = PrefillState.PREEMPTED
        req1.prefill_state = PrefillState.PREEMPTED
        req0.flowprefill_ctx.split_forward_batch = SimpleNamespace(batch_size=1)
        req1.flowprefill_ctx.split_forward_batch = SimpleNamespace(batch_size=1)
        req0.flowprefill_ctx.resume_batch = None
        req1.flowprefill_ctx.resume_batch = None
        req0.flowprefill_ctx.preempted_at = 10.0
        req1.flowprefill_ctx.preempted_at = 11.0
        regrouped_batch = make_batch([req1, req0], split_index=2)

        self.scheduler.preempted_prefill_queue.append(req0)
        self.scheduler.preempted_prefill_queue.append(req1)
        self.scheduler.current_scheduler_metrics_enabled = True

        with patch.object(
            self.scheduler,
            "_build_flowprefill_regrouped_batch",
            return_value=regrouped_batch,
        ) as build_regrouped, patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=15.0
        ):
            selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, regrouped_batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 0)
        self.assertEqual(req0.prefill_state, PrefillState.RUNNING)
        self.assertEqual(req1.prefill_state, PrefillState.RUNNING)
        build_regrouped.assert_called_once_with([req1, req0])
        self.scheduler.metrics_collector.increment_flowprefill_resumed_requests.assert_called_once_with(
            2, "request_regroup"
        )
        self.scheduler.metrics_collector.increment_flowprefill_resume_fallback.assert_not_called()
        self.scheduler.metrics_collector.observe_flowprefill_resume.assert_called_once_with(
            resume_mode="request_regroup",
            split_index=2,
            parked_duration_seconds=5.0,
        )

    def test_regroup_does_not_mix_different_split_indices(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req1 = make_req("r1", priority=4, wait_time=2.0, split_index=3)
        req0.prefill_state = PrefillState.PREEMPTED
        req1.prefill_state = PrefillState.PREEMPTED
        req0.flowprefill_ctx.split_forward_batch = SimpleNamespace(batch_size=1)
        req1.flowprefill_ctx.split_forward_batch = SimpleNamespace(batch_size=1)
        req0.flowprefill_ctx.resume_batch = None
        req1.flowprefill_ctx.resume_batch = None
        resumed_batch = make_batch([req0], split_index=2)

        self.scheduler.preempted_prefill_queue.append(req0)
        self.scheduler.preempted_prefill_queue.append(req1)
        self.scheduler.current_scheduler_metrics_enabled = True

        with patch.object(
            self.scheduler,
            "_build_flowprefill_regrouped_batch",
            return_value=resumed_batch,
        ) as build_regrouped:
            selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, resumed_batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 1)
        self.assertIs(self.scheduler.preempted_prefill_queue[0], req1)
        build_regrouped.assert_called_once_with([req0])
        self.scheduler.metrics_collector.increment_flowprefill_resume_fallback.assert_not_called()

    def test_single_preempted_request_uses_request_owned_resume_path(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        parked_batch = make_batch([req], split_index=2)
        req.prefill_state = PrefillState.PREEMPTED
        parked_batch.split_forward_batch = SimpleNamespace()
        req.sync_flowprefill_ctx_from_batch(parked_batch)
        req.flowprefill_ctx.resume_batch = None
        resumed_batch = make_batch([req], split_index=2)

        self.scheduler.preempted_prefill_queue.append(req)

        with patch.object(
            ScheduleBatch,
            "init_single_req_from_flowprefill_ctx",
            return_value=resumed_batch,
        ) as init_resume:
            selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, resumed_batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 0)
        init_resume.assert_called_once()
        self.assertEqual(req.prefill_state, PrefillState.RUNNING)

    def test_single_request_resume_records_metrics(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req.prefill_state = PrefillState.PREEMPTED
        req.flowprefill_ctx.split_forward_batch = SimpleNamespace()
        req.flowprefill_ctx.preempted_at = 10.0
        resumed_batch = make_batch([req], split_index=2)
        self.scheduler.current_scheduler_metrics_enabled = True
        self.scheduler.preempted_prefill_queue.append(req)

        with patch.object(
            ScheduleBatch,
            "init_single_req_from_flowprefill_ctx",
            return_value=resumed_batch,
        ), patch("sglang.srt.managers.scheduler.time.perf_counter", return_value=12.0):
            selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, resumed_batch)
        self.scheduler.metrics_collector.increment_flowprefill_resumed_requests.assert_called_once_with(
            1, "single_request"
        )
        self.scheduler.metrics_collector.observe_flowprefill_resume.assert_called_once_with(
            resume_mode="single_request",
            split_index=2,
            parked_duration_seconds=2.0,
        )

    def test_single_req_resume_restores_prefill_stats_from_request_ctx(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req.flowprefill_ctx.split_forward_batch = SimpleNamespace(
            input_ids=None,
            req_pool_indices=None,
            seq_lens=None,
            seq_lens_cpu=None,
            orig_seq_lens=None,
            out_cache_loc=None,
            seq_lens_sum=0,
            encoder_out_cache_loc=None,
            input_embeds=None,
            token_type_ids=None,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
        )
        req.flowprefill_ctx.prefill_stats = object()
        req.stream = False
        req.return_logprob = False
        req.grammar = None
        req.return_hidden_states = False
        req.return_routed_experts = False
        req.is_prefill_only = False
        req.multimodal_inputs = None
        req.dimensions = None
        req.extend_input_len = 1
        req.extend_logprob_start_len = 0
        req.prefix_indices = []

        with patch.object(
            ScheduleBatch, "prepare_for_split_prefill", autospec=True
        ), patch(
            "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch",
            return_value=SimpleNamespace(),
        ):
            batch = ScheduleBatch.init_single_req_from_flowprefill_ctx(
                req,
                req_to_token_pool=SimpleNamespace(device="cpu"),
                token_to_kv_pool_allocator=SimpleNamespace(),
                tree_cache=SimpleNamespace(),
                model_config=SimpleNamespace(vocab_size=32000),
                enable_overlap=False,
                spec_algorithm=None,
            )

        self.assertIs(batch.prefill_stats, req.flowprefill_ctx.prefill_stats)

    def test_single_req_resume_supports_return_logprob(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req.return_logprob = True
        req.logprob_start_len = 1
        req.origin_input_ids = [11, 12, 13, 14]
        req.fill_ids = [11, 12, 13, 14]
        req.prefix_indices = [0, 1]
        req.extend_input_len = 2
        req.extend_logprob_start_len = 2
        req.top_logprobs_num = 3
        req.token_ids_logprob = [7, 8]
        req.flowprefill_ctx.split_forward_batch = SimpleNamespace(
            input_ids=None,
            req_pool_indices=None,
            seq_lens=None,
            seq_lens_cpu=None,
            orig_seq_lens=None,
            out_cache_loc=None,
            seq_lens_sum=0,
            encoder_out_cache_loc=None,
            input_embeds=None,
            token_type_ids=None,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
        )

        with patch.object(
            ScheduleBatch, "prepare_for_split_prefill", autospec=True
        ), patch(
            "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch",
            return_value=SimpleNamespace(),
        ):
            batch = ScheduleBatch.init_single_req_from_flowprefill_ctx(
                req,
                req_to_token_pool=SimpleNamespace(device="cpu"),
                token_to_kv_pool_allocator=SimpleNamespace(),
                tree_cache=SimpleNamespace(),
                model_config=SimpleNamespace(vocab_size=32000),
                enable_overlap=False,
                spec_algorithm=None,
            )

        self.assertTrue(batch.return_logprob)
        self.assertEqual(batch.top_logprobs_nums, [3])
        self.assertEqual(batch.token_ids_logprobs, [[7, 8]])
        self.assertEqual(batch.extend_input_logprob_token_ids.tolist(), [13, 14])

    def test_fallback_batch_resume_records_reason_and_metrics(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req1 = make_req("r1", priority=4, wait_time=2.0, split_index=2)
        batch = make_batch([req0, req1], split_index=2)
        batch.split_forward_batch = SimpleNamespace()
        req0.prefill_state = PrefillState.PREEMPTED
        req1.prefill_state = PrefillState.PREEMPTED
        req0.flowprefill_ctx.preempted_at = 20.0
        req1.flowprefill_ctx.preempted_at = 21.0
        req0.sync_flowprefill_ctx_from_batch(batch)
        req1.sync_flowprefill_ctx_from_batch(batch)
        self.scheduler.current_scheduler_metrics_enabled = True
        self.scheduler.preempted_prefill_queue.append(req0)
        self.scheduler.preempted_prefill_queue.append(req1)

        with patch(
            "sglang.srt.managers.scheduler.time.perf_counter", return_value=25.0
        ):
            selected = self.scheduler._get_next_flowprefill_candidate()

        self.assertIs(selected, batch)
        self.scheduler.metrics_collector.increment_flowprefill_resume_fallback.assert_called_once_with(
            "resume batch has multiple requests"
        )
        self.scheduler.metrics_collector.increment_flowprefill_resumed_requests.assert_called_once_with(
            2, "parked_batch_fallback"
        )
        self.scheduler.metrics_collector.observe_flowprefill_resume.assert_called_once_with(
            resume_mode="parked_batch_fallback",
            split_index=2,
            parked_duration_seconds=5.0,
        )

    def test_single_req_resume_guard_rejects_multi_req_parked_batch(self):
        req0 = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req1 = make_req("r1", priority=4, wait_time=2.0, split_index=2)
        batch = make_batch([req0, req1], split_index=2)
        batch.split_forward_batch = SimpleNamespace()
        req0.sync_flowprefill_ctx_from_batch(batch)

        reason = self.scheduler._get_flowprefill_single_req_resume_guard_failure(req0)

        self.assertEqual(reason, "resume batch has multiple requests")

    def test_single_req_resume_guard_rejects_missing_forward_state(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req.flowprefill_ctx.resume_batch = None

        reason = self.scheduler._get_flowprefill_single_req_resume_guard_failure(req)

        self.assertEqual(reason, "missing split_forward_batch")

    def test_single_req_resume_guard_allows_return_logprob(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        req.return_logprob = True
        req.flowprefill_ctx.split_forward_batch = SimpleNamespace()
        req.flowprefill_ctx.resume_batch = None

        reason = self.scheduler._get_flowprefill_single_req_resume_guard_failure(req)

        self.assertIsNone(reason)

    def test_split_prefill_uses_forward_progress_for_completion(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=2)
        batch = make_batch([req], split_index=2)
        batch.split_forward_batch = SimpleNamespace(split_index=4)
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertIsNone(self.scheduler.running_split_prefill_batch)
        self.assertEqual(batch.split_index, 4)
        self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)
        self.scheduler.process_batch_result_prefill.assert_called_once()

    def test_final_split_prefill_falls_back_to_normal_prefill_processing(self):
        req = make_req("r0", priority=5, wait_time=1.0, split_index=3)
        batch = make_batch([req], split_index=4)
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertIsNone(self.scheduler.running_split_prefill_batch)
        self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)
        self.scheduler.process_batch_result_prefill.assert_called_once()
