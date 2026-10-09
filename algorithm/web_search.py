import hashlib
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests


DEFAULT_CHECKPOINTS = [0.0, 0.2, 0.4, 0.6, 0.8]
DEFAULT_EXCLUDE_KEYWORDS = ["ExLLM", "MOLLEO"]


def _config_get(config, key: str, default: Any = None) -> Any:
    value = config.get(key, default=None)
    return default if value is None else value


def _slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "web_search"


def _stable_hash(obj: Any, n: int = 16) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def _word_limit(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _contains_excluded(text: str, keywords: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(keyword).lower() in lowered for keyword in keywords if keyword)


def _extract_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```"]:
        for block in re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


class WebSearchMemory:
    """Scientific web-search prior injection for molecule optimization.

    This class intentionally does not modify ExLLM's internal experience
    summarization. It periodically refreshes a fixed external-principles block,
    which PromptTemplate prepends to proposal prompts.
    """

    def __init__(self, config, llm, seed: int, property_list: List[str]) -> None:
        self.config = config
        self.llm = llm
        self.seed = seed
        self.property_list = property_list
        self.enabled = self._is_enabled()
        self.records: List[Dict[str, Any]] = []
        self.external_principles = ""
        self.triggered_indices = set()
        self.next_checkpoint_idx = 0

        self.checkpoints = [
            float(x) for x in _config_get(config, "web_search_checkpoints", DEFAULT_CHECKPOINTS)
        ]
        self.checkpoints = sorted(set(max(0.0, min(1.0, x)) for x in self.checkpoints))
        self.queries_per_round = int(_config_get(config, "web_search_queries_per_round", 4))
        self.queries_per_round = max(2, min(4, self.queries_per_round))
        self.results_per_query = int(_config_get(config, "web_search_results_per_query", 5))
        self.max_words_per_result = int(_config_get(config, "web_search_max_words_per_result", 500))
        self.max_external_principles = int(_config_get(config, "web_search_max_external_principles", 8))
        self.timeout = float(_config_get(config, "web_search_timeout", 20))
        self.search_sleep = float(_config_get(config, "web_search_sleep", 0.2))
        self.sources = _config_get(config, "web_search_sources", ["semantic_scholar", "openalex", "pubmed", "chembl"])
        self.exclude_keywords = _config_get(config, "web_search_exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS)
        self.exclude_keywords = [str(x) for x in self.exclude_keywords]

        # 默认缓存目录放在本 run 的 save_dir 下；硬编码的 /Users/... 路径在 Linux 机器上不可创建，
        # 会让所有非分子任务的 run 在启动时崩掉（2026-09-09 kernel 任务 8 个续跑 run）。
        cache_dir = _config_get(
            config,
            "web_search_cache_dir",
            os.path.join(str(_config_get(config, "save_dir", ".")), "web_search_cache"),
        )
        model_name = str(_config_get(config, "model.name", "unknown_model")).replace(",", "/")
        suffix = str(_config_get(config, "save_suffix", ""))
        self.cache_dir = os.path.join(
            cache_dir,
            "molecules",
            _slug(model_name.replace("/", "_")),
            _slug(suffix or "default"),
            f"seed_{seed}",
        )
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:   # 缓存目录建不出来不应影响优化本身
            print(f"web_search: cannot create cache dir {self.cache_dir} ({e}); web search disabled for this run")
            self.cache_dir = None

    def _is_enabled(self) -> bool:
        if getattr(self, "cache_dir", "") is None:
            return False
        if not (
            bool(_config_get(self.config, "web_search", False))
            or bool(_config_get(self.config, "model.web_search", False))
        ):
            return False
        evaluator_path = str(_config_get(self.config, "evalutor_path", ""))
        if "molecules" not in evaluator_path:
            print("web_search is enabled but currently only supports molecule optimization; skipping.")
            return False
        return True

    def restore(self, records: List[Dict[str, Any]], external_principles: str = "") -> None:
        self.records = records or []
        self.external_principles = external_principles or ""
        self.triggered_indices = {
            int(record.get("checkpoint_idx"))
            for record in self.records
            if record.get("checkpoint_idx") is not None
        }
        self.next_checkpoint_idx = 0
        while (
            self.next_checkpoint_idx < len(self.checkpoints)
            and self.next_checkpoint_idx in self.triggered_indices
        ):
            self.next_checkpoint_idx += 1

    def maybe_update(self, mol_buffer: List[Any], budget_used: int, budget: int) -> Optional[Dict[str, Any]]:
        if not self.enabled or budget <= 0:
            return None
        while self.next_checkpoint_idx < len(self.checkpoints):
            checkpoint = self.checkpoints[self.next_checkpoint_idx]
            trigger_budget = int(round(budget * checkpoint))
            if budget_used < trigger_budget:
                return None
            record = self.run_search_event(
                checkpoint_idx=self.next_checkpoint_idx,
                checkpoint=checkpoint,
                budget_used=budget_used,
                budget=budget,
                mol_buffer=mol_buffer,
            )
            self.triggered_indices.add(self.next_checkpoint_idx)
            self.next_checkpoint_idx += 1
            return record
        return None

    def _history_digest(self, mol_buffer: List[Any], top_k: int = 25) -> Dict[str, Any]:
        items = [entry[0] for entry in mol_buffer if entry and entry[0].total is not None]
        ranked = sorted(items, key=lambda item: item.total, reverse=True)
        compact = []
        for item in ranked[:top_k]:
            compact.append(
                {
                    "candidate": item.value,
                    "total": float(item.total),
                    "property": item.property,
                }
            )
        return {
            "n_evaluated": len(items),
            "top": compact,
            "goals": self.property_list,
            "directions": _config_get(self.config, "optimization_direction", []),
        }

    def _cache_path(self, checkpoint_idx: int, checkpoint: float, history_digest: Dict[str, Any]) -> str:
        key = {
            "checkpoint_idx": checkpoint_idx,
            "checkpoint": checkpoint,
            "history_hash": _stable_hash(history_digest),
            "goals": self.property_list,
            "seed": self.seed,
        }
        return os.path.join(
            self.cache_dir,
            f"checkpoint_{checkpoint_idx}_{checkpoint:.2f}_{_stable_hash(key)}.json",
        )

    def run_search_event(
        self,
        checkpoint_idx: int,
        checkpoint: float,
        budget_used: int,
        budget: int,
        mol_buffer: List[Any],
    ) -> Dict[str, Any]:
        history_digest = self._history_digest(mol_buffer)
        cache_path = self._cache_path(checkpoint_idx, checkpoint, history_digest)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["llm_calls_used"] = 0
            self.external_principles = record.get("external_principles", "")
            self.records.append(record)
            print(f"web_search cache hit: checkpoint={checkpoint:.2f} path={cache_path}")
            return record

        print(f"web_search trigger: checkpoint={checkpoint:.2f}, budget={budget_used}/{budget}")
        queries_prompt = self._query_generation_prompt(history_digest, checkpoint, budget_used, budget)
        query_response = self.llm.chat(queries_prompt)
        queries = self._parse_queries(query_response)
        results = []
        for query in queries:
            results.extend(self.search_query(query))
        summary_prompt = self._summary_prompt(history_digest, queries, results)
        summary_response = self.llm.chat(summary_prompt)
        external_principles = self._format_external_principles(summary_response)
        record = {
            "checkpoint_idx": checkpoint_idx,
            "checkpoint": checkpoint,
            "budget_used": budget_used,
            "budget": budget,
            "history_digest": history_digest,
            "queries_prompt": queries_prompt,
            "queries_response": query_response,
            "queries": queries,
            "results": results,
            "summary_prompt": summary_prompt,
            "summary_response": summary_response,
            "external_principles": external_principles,
            "llm_calls_used": 2,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cache_path": cache_path,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.external_principles = external_principles
        self.records.append(record)
        return record

    def _query_generation_prompt(
        self,
        history_digest: Dict[str, Any],
        checkpoint: float,
        budget_used: int,
        budget: int,
    ) -> str:
        return f"""
You are helping a molecule optimization algorithm search external scientific literature.

Task: generate {self.queries_per_round} search queries for PubMed, Semantic Scholar, and ChEMBL.
The optimization objectives are {self.property_list} with directions {_config_get(self.config, "optimization_direction", [])}.
Current budget checkpoint: {checkpoint:.0%} ({budget_used}/{budget} evaluated candidates).

Recent optimization history:
{json.dumps(history_digest, ensure_ascii=False, indent=2, default=str)}

Rules:
- Return 2 to 4 diverse search queries.
- Focus on external molecular design strategies, medicinal chemistry principles, scaffolds, bioactivity, and multi-objective tradeoffs.
- Do not search for or mention ExLLM or MOLLEO.
- Do not include the benchmark paper names.
- Do not reveal a single target molecule; queries should seek reusable design strategies.

Return only JSON:
{{
  "queries": ["query 1", "query 2"]
}}
""".strip()

    def _parse_queries(self, response: str) -> List[str]:
        parsed = _extract_json(response)
        queries = []
        if isinstance(parsed, dict):
            raw = parsed.get("queries", [])
            if isinstance(raw, list):
                queries = [str(x).strip() for x in raw if str(x).strip()]
        if not queries:
            for line in str(response).splitlines():
                line = re.sub(r"^[\\s\\-*\\d.)]+", "", line).strip()
                if len(line) > 8:
                    queries.append(line)
        clean = []
        seen = set()
        for query in queries:
            if _contains_excluded(query, self.exclude_keywords):
                continue
            query = query[:240]
            if query.lower() not in seen:
                seen.add(query.lower())
                clean.append(query)
        if len(clean) < 2:
            clean.extend(
                [
                    "multi objective molecular optimization DRD2 QED kinase scaffold medicinal chemistry",
                    "drug-like molecule design scaffold hopping synthetic accessibility dopamine D2 receptor kinase inhibitors",
                ]
            )
        return clean[: self.queries_per_round]

    def search_query(self, query: str) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        if "semantic_scholar" in self.sources:
            semantic_results = self._search_semantic_scholar(query)
            if semantic_results:
                collected.extend(semantic_results)
            else:
                print(f"web_search falling back to OpenAlex for query={query!r}")
                collected.extend(self._search_openalex(query))
        elif "openalex" in self.sources:
            collected.extend(self._search_openalex(query))
        if "pubmed" in self.sources:
            collected.extend(self._search_pubmed(query))
        if "chembl" in self.sources:
            collected.extend(self._search_chembl(query))

        clean = []
        seen = set()
        for result in collected:
            text = " ".join(str(result.get(k, "")) for k in ["title", "abstract", "snippet", "url"])
            if _contains_excluded(text, self.exclude_keywords):
                continue
            key = (result.get("source"), result.get("id") or result.get("url") or result.get("title"))
            if key in seen:
                continue
            seen.add(key)
            result["query"] = query
            for field in ["abstract", "snippet"]:
                if result.get(field):
                    result[field] = _word_limit(result[field], self.max_words_per_result)
            clean.append(result)
            if len(clean) >= self.results_per_query:
                break
        return clean

    def _search_semantic_scholar(self, query: str) -> List[Dict[str, Any]]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": self.results_per_query,
            "fields": "title,abstract,url,year,authors,venue,externalIds",
        }
        headers = {}
        api_key = _config_get(self.config, "web_search_semantic_scholar_api_key", None) or os.getenv(
            "SEMANTIC_SCHOLAR_API_KEY"
        )
        if api_key:
            headers["x-api-key"] = str(api_key)
        try:
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            time.sleep(self.search_sleep)
            rows = []
            for paper in response.json().get("data", []):
                rows.append(
                    {
                        "source": "semantic_scholar",
                        "id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract") or "",
                        "url": paper.get("url"),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "authors": [a.get("name") for a in paper.get("authors", [])[:6]],
                        "external_ids": paper.get("externalIds"),
                    }
                )
            return rows
        except Exception as exc:
            print(f"web_search semantic_scholar failed for query={query!r}: {exc}")
            return []

    def _openalex_abstract(self, inverted_index: Optional[Dict[str, List[int]]]) -> str:
        if not inverted_index:
            return ""
        positions = []
        for word, idxs in inverted_index.items():
            for idx in idxs:
                positions.append((int(idx), word))
        positions.sort()
        return " ".join(word for _, word in positions)

    def _search_openalex(self, query: str) -> List[Dict[str, Any]]:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": self.results_per_query,
        }
        mailto = _config_get(self.config, "web_search_openalex_mailto", None)
        if mailto:
            params["mailto"] = mailto
        last_exc = None
        for attempt in range(1, 6):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                time.sleep(self.search_sleep)
                rows = []
                for work in response.json().get("results", [])[: self.results_per_query]:
                    ids = work.get("ids") or {}
                    location = work.get("primary_location") or {}
                    rows.append(
                        {
                            "source": "openalex",
                            "id": work.get("id"),
                            "title": work.get("title") or work.get("display_name"),
                            "abstract": self._openalex_abstract(work.get("abstract_inverted_index")),
                            "url": ids.get("doi") or location.get("landing_page_url") or work.get("id"),
                            "year": work.get("publication_year"),
                            "venue": (
                                (location.get("source") or {}).get("display_name")
                                if isinstance(location.get("source"), dict)
                                else None
                            ),
                            "doi": ids.get("doi"),
                            "pmid": ids.get("pmid"),
                        }
                    )
                return rows
            except Exception as exc:
                last_exc = exc
                print(f"web_search openalex attempt {attempt}/5 failed for query={query!r}: {exc}")
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 20))
        print(f"web_search openalex failed after 5 attempts for query={query!r}: {last_exc}; skipping this source.")
        return []

    def _search_pubmed(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": self.results_per_query,
                "sort": "relevance",
            }
            email = _config_get(self.config, "web_search_ncbi_email", None)
            if email:
                params["email"] = email
            esearch = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                timeout=self.timeout,
            )
            esearch.raise_for_status()
            ids = esearch.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []
            time.sleep(self.search_sleep)
            efetch = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
                timeout=self.timeout,
            )
            efetch.raise_for_status()
            time.sleep(self.search_sleep)
            root = ET.fromstring(efetch.text)
            rows = []
            for article in root.findall(".//PubmedArticle"):
                pmid = article.findtext(".//PMID")
                title = "".join(article.findtext(".//ArticleTitle") or "").strip()
                abstract_parts = [
                    "".join(elem.itertext()).strip()
                    for elem in article.findall(".//Abstract/AbstractText")
                ]
                abstract = " ".join(part for part in abstract_parts if part)
                journal = article.findtext(".//Journal/Title")
                year = article.findtext(".//PubDate/Year")
                rows.append(
                    {
                        "source": "pubmed",
                        "id": pmid,
                        "title": title,
                        "abstract": abstract,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "year": year,
                        "venue": journal,
                    }
                )
            return rows
        except Exception as exc:
            print(f"web_search pubmed failed for query={query!r}: {exc}")
            return []

    def _search_chembl(self, query: str) -> List[Dict[str, Any]]:
        url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
        params = {"q": query, "limit": self.results_per_query}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            time.sleep(self.search_sleep)
            rows = []
            for mol in response.json().get("molecules", [])[: self.results_per_query]:
                structures = mol.get("molecule_structures") or {}
                rows.append(
                    {
                        "source": "chembl",
                        "id": mol.get("molecule_chembl_id"),
                        "title": mol.get("pref_name") or mol.get("molecule_chembl_id"),
                        "snippet": (
                            f"ChEMBL molecule type={mol.get('molecule_type')}; "
                            f"max_phase={mol.get('max_phase')}; "
                            f"oral={mol.get('oral')}; parenteral={mol.get('parenteral')}; "
                            f"canonical_smiles={structures.get('canonical_smiles')}"
                        ),
                        "url": (
                            "https://www.ebi.ac.uk/chembl/compound_report_card/"
                            f"{mol.get('molecule_chembl_id')}/"
                            if mol.get("molecule_chembl_id")
                            else None
                        ),
                    }
                )
            return rows
        except Exception as exc:
            print(f"web_search chembl failed for query={query!r}: {exc}")
            return []

    def _summary_prompt(
        self,
        history_digest: Dict[str, Any],
        queries: List[str],
        results: List[Dict[str, Any]],
    ) -> str:
        compact_results = []
        for result in results:
            compact_results.append(
                {
                    "source": result.get("source"),
                    "title": result.get("title"),
                    "year": result.get("year"),
                    "venue": result.get("venue"),
                    "url": result.get("url"),
                    "abstract_or_snippet": result.get("abstract") or result.get("snippet"),
                }
            )
        return f"""
You are extracting reusable external design principles for molecule optimization.

Optimization objectives: {self.property_list}
Optimization directions: {_config_get(self.config, "optimization_direction", [])}

Current optimization history:
{json.dumps(history_digest, ensure_ascii=False, indent=2, default=str)}

Search queries:
{json.dumps(queries, ensure_ascii=False, indent=2)}

Search results:
{json.dumps(compact_results, ensure_ascii=False, indent=2, default=str)}

Rules:
- Produce at most {self.max_external_principles} concise external principles/strategies.
- Keep the result fixed length and non-repetitive.
- Use only general external scientific strategies; do not reveal an exact target molecule.
- Do not mention ExLLM or MOLLEO.
- Prefer strategies that could increase scaffold/principle diversity rather than tiny substituent edits.
- Explicitly mark uncertainty if evidence is indirect.

Return only JSON:
{{
  "external_principles": [
    "strategy 1",
    "strategy 2"
  ]
}}
""".strip()

    def _format_external_principles(self, response: str) -> str:
        parsed = _extract_json(response)
        principles = []
        if isinstance(parsed, dict) and isinstance(parsed.get("external_principles"), list):
            principles = [str(x).strip() for x in parsed["external_principles"] if str(x).strip()]
        if not principles:
            for line in str(response).splitlines():
                line = re.sub(r"^[\\s\\-*\\d.)]+", "", line).strip()
                if len(line) > 20 and not _contains_excluded(line, self.exclude_keywords):
                    principles.append(line)
        clean = []
        seen = set()
        for principle in principles:
            if _contains_excluded(principle, self.exclude_keywords):
                continue
            principle = _word_limit(principle, 80)
            key = principle.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(principle)
        clean = clean[: self.max_external_principles]
        if not clean:
            return ""
        body = "\n".join(f"- {principle}" for principle in clean)
        return (
            "External literature/database strategies are available. "
            "Use them as broad search directions, not as fixed candidate answers:\n"
            f"<external_principles>\n{body}\n</external_principles>\n"
        )
