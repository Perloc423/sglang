from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.schedule_batch import PrefillState
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase


def make_req(rid: str, priority: int, wait_time: float, split_index: int = 0):
    flowprefill_ctx = SimpleNamespace(
        split_index=split_index,
        split_forward_batch=None,
        seq_lens_cpu_cache=None,
        split_attn_backend_needs_reinit=False,
        resume_batch=None,
    )
    req = SimpleNamespace()
    req.rid = rid
    req.priority = priority
    req.prefill_deadline_ts = None
    req.prefill_preempt_pending = False
    req.prefill_num_preemptions = 0
    req.prefill_resume_split_index = split_index
    req.prefill_state = PrefillState.WAITING
    req.to_finish = None
    req.finished = lambda: False
    req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
    req.flowprefill_ctx = flowprefill_ctx

    def sync_flowprefill_ctx_from_batch(batch):
        req.prefill_resume_split_index = batch.split_index
        req.flowprefill_ctx.split_index = batch.split_index
        req.flowprefill_ctx.split_forward_batch = getattr(
            batch, "split_forward_batch", None
        )
        req.flowprefill_ctx.seq_lens_cpu_cache = getattr(batch, "seq_lens_cpu_cache", None)
        req.flowprefill_ctx.split_attn_backend_needs_reinit = getattr(
            batch, "split_attn_backend_needs_reinit", False
        )
        req.flowprefill_ctx.resume_batch = batch

    def apply_flowprefill_ctx_to_batch(batch):
        batch.split_index = req.flowprefill_ctx.split_index
        batch.split_forward_batch = req.flowprefill_ctx.split_forward_batch
        batch.seq_lens_cpu_cache = req.flowprefill_ctx.seq_lens_cpu_cache
        batch.split_attn_backend_needs_reinit = (
            req.flowprefill_ctx.split_attn_backend_needs_reinit
        )

    def reset_flowprefill_ctx():
        req.flowprefill_ctx.split_index = 0
        req.flowprefill_ctx.split_forward_batch = None
        req.flowprefill_ctx.seq_lens_cpu_cache = None
        req.flowprefill_ctx.split_attn_backend_needs_reinit = False
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
        )
        self.scheduler.schedule_low_priority_values_first = False
        self.scheduler.enable_flowprefill = True
        self.scheduler.policy = MagicMock()
        self.scheduler.policy.calc_priority.side_effect = (
            lambda waiting_queue, running_batch: waiting_queue.sort(
                key=self.scheduler._flowprefill_priority_key
            )
        )
        self.scheduler.running_batch = SimpleNamespace(reqs=[])
        self.scheduler.preempted_prefill_queue = deque()
        self.scheduler.running_split_prefill_batch = None
        self.scheduler.tree_cache = MagicMock()
        self.scheduler.model_config = SimpleNamespace(num_hidden_layers=4)
        self.scheduler.process_batch_result_prefill = MagicMock()

    def test_arrival_marks_running_batch_preempt_pending(self):
        running_req = make_req("running", priority=5, wait_time=10.0)
        running_batch = make_batch([running_req], split_index=1)
        self.scheduler.running_split_prefill_batch = running_batch

        incoming_req = make_req("incoming", priority=10, wait_time=20.0)
        self.scheduler._maybe_mark_flowprefill_preempt_pending(incoming_req)

        self.assertTrue(running_req.prefill_preempt_pending)
        self.assertEqual(running_req.prefill_state, PrefillState.PREEMPT_PENDING)

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
        self.scheduler.running_split_prefill_batch = batch

        self.scheduler.process_batch_result_split_prefill(batch, MagicMock())

        self.assertIsNone(self.scheduler.running_split_prefill_batch)
        self.assertEqual(len(self.scheduler.preempted_prefill_queue), 1)
        self.assertEqual(req.prefill_state, PrefillState.PREEMPTED)
        self.assertEqual(req.prefill_resume_split_index, 2)

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
