"""Internet Learning Engine with robots.txt compliance and polite crawling."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from ai_genesis.config import InternetConfig


@dataclass(slots=True)
class LearnedDocument:
    url: str
    title: str
    text: str


class InternetLearningEngine:
    """Downloads educational text without aggressive parsing."""

    def __init__(self, config: InternetConfig | None = None) -> None:
        self.config = config or InternetConfig()
        self._last_request_at = 0.0

    def collect(self, urls: Iterable[str]) -> list[LearnedDocument]:
        documents: list[LearnedDocument] = []
        for url in list(urls)[: self.config.max_pages_per_run]:
            if not self.allowed_by_robots(url):
                continue
            documents.append(self.fetch_article(url))
        return documents

    def allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser(robots_url)
        parser.read()
        return parser.can_fetch(self.config.user_agent, url)

    def fetch_article(self, url: str) -> LearnedDocument:
        self._polite_delay()
        requests = importlib.import_module("requests")
        bs4 = importlib.import_module("bs4")
        response = requests.get(
            url,
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        paragraphs = [node.get_text(" ", strip=True) for node in soup.find_all(["p", "li", "article", "section"])]
        text = "\n".join(part for part in paragraphs if len(part) > 30)
        return LearnedDocument(url=url, title=title, text=text)

    def _polite_delay(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.min_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()
