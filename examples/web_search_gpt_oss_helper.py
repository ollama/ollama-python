from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlparse

from ollama import Client


@dataclass
class Page:
  url: str
  title: str
  text: str
  lines: List[str]
  links: Dict[int, str]
  fetched_at: datetime


@dataclass
class BrowserStateData:
  page_stack: List[str] = field(default_factory=list)
  view_tokens: int = 1024
  url_to_page: Dict[str, Page] = field(default_factory=dict)


@dataclass
class WebSearchResult:
  title: str
  url: str
  content: Dict[str, str]


class SearchClient(Protocol):
  def search(self, queries: List[str], max_results: Optional[int] = None): ...


class CrawlClient(Protocol):
  def crawl(self, urls: List[str]): ...


# ---- Constants ---------------------------------------------------------------

DEFAULT_VIEW_TOKENS = 1024
CAPPED_TOOL_CONTENT_LEN = 8000

# ---- Helpers ----------------------------------------------------------------


def cap_tool_content(text: str) -> str:
  if not text:
    return text
  if len(text) <= CAPPED_TOOL_CONTENT_LEN:
    return text
  if CAPPED_TOOL_CONTENT_LEN <= 1:
    return text[:CAPPED_TOOL_CONTENT_LEN]
  return text[: CAPPED_TOOL_CONTENT_LEN - 1] + '…'


def _safe_domain(u: str) -> str:
  try:
    parsed = urlparse(u)
    host = parsed.netloc or u
    return host.replace('www.', '') if host else u
  except Exception:
    return u


# ---- BrowserState ------------------------------------------------------------


class BrowserState:
  def __init__(self, initial_state: Optional[BrowserStateData] = None):
    self._data = initial_state or BrowserStateData(view_tokens=DEFAULT_VIEW_TOKENS)

  def get_data(self) -> BrowserStateData:
    return self._data

  def set_data(self, data: BrowserStateData) -> None:
    self._data = data


# ---- Browser ----------------------------------------------------------------


class Browser:
  def __init__(
    self,
    initial_state: Optional[BrowserStateData] = None,
    client: Optional[Client] = None,
  ):
    self.state = BrowserState(initial_state)
    self._client: Optional[Client] = client

  def set_client(self, client: Client) -> None:
    self._client = client

  def get_state(self) -> BrowserStateData:
    return self.state.get_data()

  # ---- internal utils ----

  def _save_page(self, page: Page) -> None:
    data = self.state.get_data()
    data.url_to_page[page.url] = page
    data.page_stack.append(page.url)
    self.state.set_data(data)

  def _page_from_stack(self, url: str) -> Page:
    data = self.state.get_data()
    page = data.url_to_page.get(url)
    if not page:
      raise ValueError(f'Page not found for url {url}')
    return page

  def _join_lines_with_numbers(self, lines: List[str]) -> str:
    result = []
    for i, line in enumerate(lines):
      result.append(f'L{i}: {line}')
    return '\n'.join(result)

  def _wrap_lines(self, text: str, width: int = 80) -> List[str]:
    if width <= 0:
      width = 80
    src_lines = text.split('\n')
    wrapped: List[str] = []
    for line in src_lines:
      if line == '':
        wrapped.append('')
      elif len(line) <= width:
        wrapped.append(line)
      else:
        words = re.split(r'\s+', line)
        if not words:
          wrapped.append(line)
          continue
        curr = ''
        for w in words:
          test = (curr + ' ' + w) if curr else w
          if len(test) > width and curr:
            wrapped.append(curr)
            curr = w
          else:
            curr = test
        if curr:
          wrapped.append(curr)
    return wrapped

  def _process_markdown_links(self, text: str) -> Tuple[str, Dict[int, str]]:
    links: Dict[int, str] = {}
    link_id = 0

    multiline_pattern = re.compile(r'\[([^\]]+)\]\s*\n\s*\(([^)]+)\)')
    text = multiline_pattern.sub(lambda m: f'[{m.group(1)}]({m.group(2)})', text)
    text = re.sub(r'\s+', ' ', text)

    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    def _repl(m: re.Match) -> str:
      nonlocal link_id
      link_text = m.group(1).strip()
      link_url = m.group(2).strip()
      domain = _safe_domain(link_url)
      formatted = f'【{link_id}†{link_text}†{domain}】'
      links[link_id] = link_url
      link_id += 1
      return formatted

    processed = link_pattern.sub(_repl, text)
    return processed, links

  def _get_end_loc(self, loc: int, num_lines: int, total_lines: int, lines: List[str]) -> int:
    if num_lines <= 0:
      txt = self._join_lines_with_numbers(lines[loc:])
      data = self.state.get_data()
      chars_per_token = 4
      max_chars = min(data.view_tokens * chars_per_token, len(txt))
      num_lines = txt[:max_chars].count('\n') + 1
    return min(loc + num_lines, total_lines)

  def _display_page(self, page: Page, cursor: int, loc: int, num_lines: int) -> str:
    total_lines = len(page.lines) or 0
    if total_lines == 0:
      page.lines = ['']
      total_lines = 1

    if loc != loc or loc < 0:
      loc = 0
    elif loc >= total_lines:
      loc = max(0, total_lines - 1)

    end_loc = self._get_end_loc(loc, num_lines, total_lines, page.lines)

    header = f'[{cursor}] {page.title}'
    header += f'({page.url})\n' if page.url else '\n'
    header += f'**viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}**\n\n'

    body_lines = []
    for i in range(loc, end_loc):
      body_lines.append(f'L{i}: {page.lines[i]}')

    return header + '\n'.join(body_lines)

  # ---- page builders ----

  def _build_search_results_page_collection(self, query: str, results: Dict[str, Any]) -> Page:
    page = Page(
      url=f'search_results_{query}',
      title=query,
      text='',
      lines=[],
      links={},
      fetched_at=datetime.utcnow(),
    )

    tb = []
    tb.append('')
    tb.append('# Search Results')
    tb.append('')

    link_idx = 0
    for query_results in results.get('results', {}).values():
      for result in query_results:
        domain = _safe_domain(result.get('url', ''))
        link_fmt = f'* 【{link_idx}†{result.get("title", "")}†{domain}】'
        tb.append(link_fmt)

        raw_snip = result.get('content') or ''
        capped = (raw_snip[:400] + '…') if len(raw_snip) > 400 else raw_snip
        cleaned = re.sub(r'\d{40,}', lambda m: m.group(0)[:40] + '…', capped)
        cleaned = re.sub(r'\s{3,}', ' ', cleaned)
        tb.append(cleaned)
        page.links[link_idx] = result.get('url', '')
        link_idx += 1

    page.text = '\n'.join(tb)
    page.lines = self._wrap_lines(page.text, 80)
    return page

  def _build_search_result_page(self, result: WebSearchResult, link_idx: int) -> Page:
    page = Page(
      url=result.url,
      title=result.title,
      text='',
      lines=[],
      links={},
      fetched_at=datetime.utcnow(),
    )

    link_fmt = f'【{link_idx}†{result.title}】\n'
    preview = link_fmt + f'URL: {result.url}\n'
    full_text = result.content.get('fullText', '') if result.content else ''
    preview += full_text[:300] + '\n\n'

    if not full_text:
      page.links[link_idx] = result.url

    if full_text:
      raw = f'URL: {result.url}\n{full_text}'
      processed, links = self._process_markdown_links(raw)
      page.text = processed
      page.links = links
    else:
      page.text = preview

    page.lines = self._wrap_lines(page.text, 80)
    return page

  def _build_page_from_fetch(self, requested_url: str, fetch_response: Dict[str, Any]) -> Page:
    page = Page(
      url=requested_url,
      title=requested_url,
      text='',
      lines=[],
      links={},
      fetched_at=datetime.utcnow(),
    )

    for url, url_results in fetch_response.get('results', {}).items():
      if url_results:
        r0 = url_results[0]
        if r0.get('content'):
          page.text = r0['content']
        if r0.get('title'):
          page.title = r0['title']
        page.url = url
        break

    if not page.text:
      page.text = 'No content could be extracted from this page.'
    else:
      page.text = f'URL: {page.url}\n{page.text}'

    processed, links = self._process_markdown_links(page.text)
    page.text = processed
    page.links = links
    page.lines = self._wrap_lines(page.text, 80)
    return page

  def _build_find_results_page(self, pattern: str, page: Page) -> Page:
    find_page = Page(
      url=f'find_results_{pattern}',
      title=f'Find results for text: `{pattern}` in `{page.title}`',
      text='',
      lines=[],
      links={},
      fetched_at=datetime.utcnow(),
    )

    max_results = 50
    num_show_lines = 4
    pattern_lower = pattern.lower()

    result_chunks: List[str] = []
    line_idx = 0
    while line_idx < len(page.lines):
      line = page.lines[line_idx]
      if pattern_lower not in line.lower():
        line_idx += 1
        continue

      end_line = min(line_idx + num_show_lines, len(page.lines))
      snippet = '\n'.join(page.lines[line_idx:end_line])
      link_fmt = f'【{len(result_chunks)}†match at L{line_idx}】'
      result_chunks.append(f'{link_fmt}\n{snippet}')

      if len(result_chunks) >= max_results:
        break
      line_idx += num_show_lines

    if not result_chunks:
      find_page.text = f'No `find` results for pattern: `{pattern}`'
    else:
      find_page.text = '\n\n'.join(result_chunks)

    find_page.lines = self._wrap_lines(find_page.text, 80)
    return find_page

  # ---- public API: search / open / find ------------------------------------

  def search(self, *, query: str, topn: int = 5) -> Dict[str, Any]:
    if not self._client:
      raise RuntimeError('Client not provided')

    resp = self._client.web_search(query, max_results=topn)

    normalized: Dict[str, Any] = {'results': {}}
    rows: List[Dict[str, str]] = []
    for item in resp.results:
      content = item.content or ''
      rows.append(
        {
          'title': item.title,
          'url': item.url,
          'content': content,
        }
      )
    normalized['results'][query] = rows

    search_page = self._build_search_results_page_collection(query, normalized)
    self._save_page(search_page)
    cursor = len(self.get_state().page_stack) - 1

    for query_results in normalized.get('results', {}).values():
      for i, r in enumerate(query_results):
        ws = WebSearchResult(
          title=r.get('title', ''),
          url=r.get('url', ''),
          content={'fullText': r.get('content', '') or ''},
        )
        result_page = self._build_search_result_page(ws, i + 1)
        data = self.get_state()
        data.url_to_page[result_page.url] = result_page
        self.state.set_data(data)

    page_text = self._display_page(search_page, cursor, loc=0, num_lines=-1)
    return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

  def open(
    self,
    *,
    id: Optional[str | int] = None,
    cursor: int = -1,
    loc: int = 0,
    num_lines: int = -1,
  ) -> Dict[str, Any]:
    if not self._client:
      raise RuntimeError('Client not provided')

    state = self.get_state()

    if isinstance(id, str):
      url = id
      if url in state.url_to_page:
        self._save_page(state.url_to_page[url])
        cursor = len(self.get_state().page_stack) - 1
        page_text = self._display_page(state.url_to_page[url], cursor, loc, num_lines)
        return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

      fetch_response = self._client.web_fetch(url)
      normalized: Dict[str, Any] = {
        'results': {
          url: [
            {
              'title': fetch_response.title or url,
              'url': url,
              'content': fetch_response.content or '',
            }
          ]
        }
      }
      new_page = self._build_page_from_fetch(url, normalized)
      self._save_page(new_page)
      cursor = len(self.get_state().page_stack) - 1
      page_text = self._display_page(new_page, cursor, loc, num_lines)
      return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

    # Resolve current page from stack only if needed (int id or no id)
    page: Optional[Page] = None
    if cursor >= 0:
      if state.page_stack:
        if cursor >= len(state.page_stack):
          cursor = max(0, len(state.page_stack) - 1)
        page = self._page_from_stack(state.page_stack[cursor])
      else:
        page = None
    else:
      if state.page_stack:
        page = self._page_from_stack(state.page_stack[-1])

    if isinstance(id, int):
      if not page:
        raise RuntimeError('No current page to resolve link from')

      link_url = page.links.get(id)
      if not link_url:
        err = Page(
          url=f'invalid_link_{id}',
          title=f'No link with id {id} on `{page.title}`',
          text='',
          lines=[],
          links={},
          fetched_at=datetime.utcnow(),
        )
        available = sorted(page.links.keys())
        available_list = ', '.join(map(str, available)) if available else '(none)'
        err.text = '\n'.join(
          [
            f'Requested link id: {id}',
            f'Current page: {page.title}',
            f'Available link ids on this page: {available_list}',
            '',
            'Tips:',
            '- To scroll this page, call browser_open with { loc, num_lines } (no id).',
            '- To open a result from a search results page, pass the correct { cursor, id }.',
          ]
        )
        err.lines = self._wrap_lines(err.text, 80)
        self._save_page(err)
        cursor = len(self.get_state().page_stack) - 1
        page_text = self._display_page(err, cursor, 0, -1)
        return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

      new_page = state.url_to_page.get(link_url)
      if not new_page:
        fetch_response = self._client.web_fetch(link_url)
        normalized: Dict[str, Any] = {
          'results': {
            link_url: [
              {
                'title': fetch_response.title or link_url,
                'url': link_url,
                'content': fetch_response.content or '',
              }
            ]
          }
        }
        new_page = self._build_page_from_fetch(link_url, normalized)

      self._save_page(new_page)
      cursor = len(self.get_state().page_stack) - 1
      page_text = self._display_page(new_page, cursor, loc, num_lines)
      return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

    if not page:
      raise RuntimeError('No current page to display')

    cur = self.get_state()
    cur.page_stack.append(page.url)
    self.state.set_data(cur)
    cursor = len(cur.page_stack) - 1
    page_text = self._display_page(page, cursor, loc, num_lines)
    return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}

  def find(self, *, pattern: str, cursor: int = -1) -> Dict[str, Any]:
    state = self.get_state()
    if cursor == -1:
      if not state.page_stack:
        raise RuntimeError('No pages to search in')
      page = self._page_from_stack(state.page_stack[-1])
      cursor = len(state.page_stack) - 1
    else:
      if cursor < 0 or cursor >= len(state.page_stack):
        cursor = max(0, min(cursor, len(state.page_stack) - 1))
      page = self._page_from_stack(state.page_stack[cursor])

    find_page = self._build_find_results_page(pattern, page)
    self._save_page(find_page)
    new_cursor = len(self.get_state().page_stack) - 1

    page_text = self._display_page(find_page, new_cursor, 0, -1)
    return {'state': self.get_state(), 'pageText': cap_tool_content(page_text)}
