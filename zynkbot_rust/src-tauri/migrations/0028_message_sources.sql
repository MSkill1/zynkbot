-- 0028: the sources behind a searched or page-read answer stay with the message
-- (2026-10-30). They were page state only: the "View n search sources" list vanished on
-- restart while the answer itself stayed. JSON in the shape the page renders:
-- { "query": "...", "results": [ { "title", "url", "snippet" } ], "num_results": n }
ALTER TABLE conversation_messages ADD COLUMN sources_json TEXT;
