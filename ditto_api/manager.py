import asyncio
import contextlib
import logging
import shutil
import time
import uuid

from .media import prepare_portrait
from .session import Session

log = logging.getLogger(__name__)


class SessionBusy(Exception):
    pass


class Manager:
    def __init__(self, settings, engine):
        self.settings, self.engine = settings, engine
        self.sessions = {}
        self.lock = asyncio.Lock()
        self.active = None
        self.model_load_count = 1
        self.reaper = None

    async def create(self, avatar, resolution, motion, mouth):
        if self.lock.locked():
            raise SessionBusy("One stream is already active. Delete it before creating another.")
        await self.lock.acquire()
        folder = self.settings.runtime / uuid.uuid4().hex
        s = None
        try:
            folder.mkdir(parents=True)
            width, height = await asyncio.to_thread(prepare_portrait, avatar, resolution, folder / "portrait.png")
            s = Session(self, folder, resolution, width, height, motion, mouth)
            self.active = s
            await asyncio.to_thread(self.engine.setup, s)
            self.sessions[s.id] = s
            return s
        except BaseException:
            if s:
                await s.close()
            else:
                if self.lock.locked():
                    self.lock.release()
                shutil.rmtree(folder, ignore_errors=True)
            raise

    def release(self, s):
        self.sessions.pop(s.id, None)
        if self.active is s:
            self.active = None
            if self.lock.locked():
                self.lock.release()
        shutil.rmtree(s.folder, ignore_errors=True)

    async def cleanup(self):
        while True:
            await asyncio.sleep(5)
            for s in list(self.sessions.values()):
                age = time.monotonic() - s.created
                if (
                    s.error
                    or age > self.settings.max_session_seconds
                    or (not s.pc and age > self.settings.connect_timeout)
                ):
                    try:
                        await s.close()
                    except Exception:
                        log.exception("Failed to close expired session")

    async def close(self):
        if self.reaper:
            self.reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reaper
        for s in list(self.sessions.values()):
            await s.close()
