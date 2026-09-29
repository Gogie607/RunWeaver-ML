from __future__ import annotations

from queue import Full, Queue
from threading import Event, Thread
from typing import Iterator


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

    # -------------------------------------------------

    def _fetch_loop(
        self,
        raw_queue: Queue,
        stop_event: Event,
    ) -> None:

        try:
            for sample in self.pipeline.iter_root():

                if stop_event.is_set():
                    break

                if not self._put(
                    raw_queue,
                    sample,
                    stop_event,
                ):
                    break

        except BaseException as exc:
            self._put(
                raw_queue,
                exc,
                stop_event,
            )


        finally:
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

                item = raw_queue.get()

                if item is _END:
                    break

                if isinstance(item, BaseException):
                    raise item

                sample = self.pipeline.process_sample(
                    item
                )

                if sample is not None:
                    if not self._put(
                        ready_queue,
                        sample,
                        stop_event,
                    ):
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