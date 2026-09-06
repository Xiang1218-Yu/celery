"""Event receiver implementation."""
import time
from operator import itemgetter

from kombu import Queue
from kombu.connection import maybe_channel
from kombu.mixins import ConsumerMixin

from celery import uuid
from celery.app import app_or_default
from celery.exceptions import ImproperlyConfigured
from celery.utils.time import adjust_timestamp

from .event import get_exchange

__all__ = ('EventReceiver',)

CLIENT_CLOCK_SKEW = -1

_TZGETTER = itemgetter('utcoffset', 'timestamp')


class EventReceiver(ConsumerMixin):
    """Capture events.

    Arguments:
        channel (kombu.Channel): Channel to consume events on. A
            :class:`kombu.Connection` is also accepted, in which case its
            default channel is used.
        handlers (Mapping[Callable]): Event handlers.
            This is  a map of event type names and their handlers.
            The special handler `"*"` captures all events that don't have a
            handler.
        journal (celery.events.journal.EventJournal): Optional persistent
            event journal. When set, every received event is appended to
            the journal before being dispatched, so the event stream can
            be replayed after a restart/reconnect via :meth:`catchup`.
            When ``None`` (the default) monitoring behaves exactly as
            before.
    """

    app = None

    def __init__(self, channel, handlers=None, routing_key='#',
                 node_id=None, app=None, queue_prefix=None,
                 accept=None, queue_ttl=None, queue_expires=None,
                 queue_exclusive=None,
                 queue_durable=None, journal=None):
        self.app = app_or_default(app or self.app)
        self.channel = maybe_channel(channel)
        self.handlers = {} if handlers is None else handlers
        self.routing_key = routing_key
        self.node_id = node_id or uuid()
        self.queue_prefix = queue_prefix or self.app.conf.event_queue_prefix
        self.exchange = get_exchange(
            self.connection or self.app.connection_for_write(),
            name=self.app.conf.event_exchange)
        if queue_ttl is None:
            queue_ttl = self.app.conf.event_queue_ttl
        if queue_expires is None:
            queue_expires = self.app.conf.event_queue_expires
        if queue_exclusive is None:
            queue_exclusive = self.app.conf.event_queue_exclusive
        if queue_durable is None:
            queue_durable = self.app.conf.event_queue_durable
        if queue_exclusive and queue_durable:
            raise ImproperlyConfigured(
                'Queue cannot be both exclusive and durable, '
                'choose one or the other.'
            )
        self.queue = Queue(
            '.'.join([self.queue_prefix, self.node_id]),
            exchange=self.exchange,
            routing_key=self.routing_key,
            auto_delete=not queue_durable,
            durable=queue_durable,
            exclusive=queue_exclusive,
            message_ttl=queue_ttl,
            expires=queue_expires,
        )
        self.clock = self.app.clock
        self.adjust_clock = self.clock.adjust
        self.forward_clock = self.clock.forward
        if accept is None:
            accept = {self.app.conf.event_serializer, 'json'}
        self.accept = accept
        self.journal = journal

    def process(self, type, event):
        """Process event: persist to journal (if enabled), then dispatch."""
        if self.journal is not None:
            # Append first so an event is durable even if a handler raises;
            # replay then provides at-least-once delivery after a crash.
            self.journal.append(event)
        self.dispatch(type, event)

    def dispatch(self, type, event):
        """Dispatch event to the configured handler without journaling it."""
        handler = self.handlers.get(type) or self.handlers.get('*')
        handler and handler(event)

    def catchup(self, after_cursor=0, limit=None, **filters):
        """Replay journaled events to the configured handlers.

        Entries with ``seq > after_cursor`` are dispatched exactly as live
        events would be (without being appended to the journal again), and
        the cursor of the last replayed entry is returned. Pass that value
        back in after a restart/reconnect to resume the stream with no
        gaps or duplicates. Supports the same filters as
        :meth:`celery.events.journal.EventJournal.read`.

        Returns *after_cursor* unchanged when no journal is configured.
        """
        if self.journal is None:
            return after_cursor
        last_cursor = after_cursor
        for entry in self.journal.replay(
                after_cursor=after_cursor, limit=limit, **filters):
            self.dispatch(entry.type, entry.event)
            last_cursor = entry.seq
        return last_cursor

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=[self.queue],
                         callbacks=[self._receive], no_ack=True,
                         accept=self.accept)]

    def on_consume_ready(self, connection, channel, consumers,
                         wakeup=True, **kwargs):
        if wakeup:
            self.wakeup_workers(channel=channel)

    def itercapture(self, limit=None, timeout=None, wakeup=True):
        return self.consume(limit=limit, timeout=timeout, wakeup=wakeup)

    def capture(self, limit=None, timeout=None, wakeup=True):
        """Open up a consumer capturing events.

        This has to run in the main process, and it will never stop
        unless :attr:`EventDispatcher.should_stop` is set to True, or
        forced via :exc:`KeyboardInterrupt` or :exc:`SystemExit`.
        """
        for _ in self.consume(limit=limit, timeout=timeout, wakeup=wakeup):
            pass

    def wakeup_workers(self, channel=None):
        self.app.control.broadcast('heartbeat',
                                   connection=self.connection,
                                   channel=channel)

    def event_from_message(self, body, localize=True,
                           now=time.time, tzfields=_TZGETTER,
                           adjust_timestamp=adjust_timestamp,
                           CLIENT_CLOCK_SKEW=CLIENT_CLOCK_SKEW):
        type = body['type']
        if type == 'task-sent':
            # clients never sync so cannot use their clock value
            _c = body['clock'] = (self.clock.value or 1) + CLIENT_CLOCK_SKEW
            self.adjust_clock(_c)
        else:
            try:
                clock = body['clock']
            except KeyError:
                body['clock'] = self.forward_clock()
            else:
                self.adjust_clock(clock)

        if localize:
            try:
                offset, timestamp = tzfields(body)
            except KeyError:
                pass
            else:
                body['timestamp'] = adjust_timestamp(timestamp, offset)
        body['local_received'] = now()
        return type, body

    def _receive(self, body, message, list=list, isinstance=isinstance):
        if isinstance(body, list):  # celery 4.0+: List of events
            process, from_message = self.process, self.event_from_message
            [process(*from_message(event)) for event in body]
        else:
            self.process(*self.event_from_message(body))

    @property
    def connection(self):
        return self.channel.connection.client if self.channel else None
