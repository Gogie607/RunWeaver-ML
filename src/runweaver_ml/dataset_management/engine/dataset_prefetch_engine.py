from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Iterator

from .transport.transport_base import DeferredPayload


_END = object()
_WORKER_END = object()

class DatasetPrefetchEngine:

    def __init__(
        self,
        pipeline,
        *,
        queue_size: int = 32,
        name: str = "dataset-prefetch",
        num_workers: int = 1,
    ):
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")

        self.pipeline = pipeline
        self.queue_size = queue_size
        self.name = name
        self.num_workers = num_workers

    # -------------------------------------------------

    def __iter__(self) -> Iterator[dict]:

        raw_queue = Queue(maxsize=self.queue_size)
        ready_queue = Queue(maxsize=self.queue_size)
        stop_event = Event()

        fetcher = Thread(
            target=self._fetch_loop,
            args=(raw_queue, stop_event),
            name=f"{self.name}-fetch",
            daemon=True,
        )

        workers = []

        for i in range(self.num_workers):
            workers.append(
                Thread(
                    target=self._worker_loop,
                    args=(
                        raw_queue,
                        ready_queue,
                        stop_event,
                    ),
                    name=f"{self.name}-worker-{i}",
                    daemon=True,
                )
            )

        for w in workers:
            w.start()

        fetcher.start()
        finished_workers = 0

        try:
            while finished_workers < self.num_workers:
                item = ready_queue.get()

                if item is _WORKER_END:
                    finished_workers += 1
                    continue

                if isinstance(item, BaseException):
                    raise item

                yield item

        finally:
            stop_event.set()

            fetcher.join(timeout=1.0)

            for worker in workers:
                worker.join(timeout=1.0)

            # Retained exception tracebacks may keep this frame and its queues
            # alive. Explicitly abandon samples that will never be delivered.
            for queue in (raw_queue, ready_queue):
                while True:
                    try:
                        item = queue.get_nowait()
                    except Empty:
                        break
                    self._close_deferred_sample(item)

    # -------------------------------------------------

    def _fetch_loop(
        self,
        raw_queue: Queue,
        stop_event: Event,
    ) -> None:

        root_iterator = None
        try:
            root_iterator = self.pipeline.iter_root()
            for sample in root_iterator:

                if stop_event.is_set():
                    self._close_deferred_sample(sample)
                    break

                if not self._put(
                    raw_queue,
                    sample,
                    stop_event,
                ):
                    self._close_deferred_sample(sample)
                    break

        except BaseException as exc:
            self._put(
                raw_queue,
                exc,
                stop_event,
            )


        finally:
            try:
                close = getattr(root_iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as exc:
                self._put(raw_queue, exc, stop_event)
            finally:
                root_iterator = None
            for _ in range(self.num_workers):
                self._put(
                    raw_queue,
                    _END,
                    stop_event,
                )

    # -------------------------------------------------

    def _worker_loop(
        self,
        raw_queue: Queue,
        ready_queue: Queue,
        stop_event: Event,
    ) -> None:

        try:
            while not stop_event.is_set():

                try:
                    item = raw_queue.get(timeout=0.1)
                except Empty:
                    continue

                if item is _END:
                    break

                if isinstance(item, BaseException):
                    raise item

                try:
                    sample = self.pipeline.process_sample(item)
                except BaseException:
                    self._close_deferred_sample(item)
                    raise

                if sample is not None:
                    if not self._put(
                        ready_queue,
                        sample,
                        stop_event,
                    ):
                        self._close_deferred_sample(sample)
                        break

        except BaseException as exc:
            self._put(
                ready_queue,
                exc,
                stop_event,
            )

        finally:
            self._put(
                ready_queue,
                _WORKER_END,
                stop_event,
            )

    # -------------------------------------------------

    @staticmethod
    def _close_deferred_sample(sample):
        if isinstance(sample, dict):
            for value in sample.values():
                if isinstance(value, DeferredPayload):
                    value.close()

    @staticmethod
    def _put(
        queue: Queue,
        item,
        stop_event: Event,
    ) -> bool:

        while not stop_event.is_set():
            try:
                queue.put(
                    item,
                    timeout=0.1,
                )
                return True

            except Full:
                continue

        return False