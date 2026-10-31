//! Reading a page the user points at (Matt, 2026-10-27). A web address in the message
//! is the signal: the page is fetched on this device, its readable text is pulled out,
//! and the text goes into the prompt the way a Knowledge Base passage does, so every
//! backend can read it, local models included. A search engine returns its index; this
//! returns the page in front of the user.
use std::time::Duration;

/// A page read for the prompt.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReadPage {
    pub url: String,
    pub title: String,
    pub text: String,
    /// Characters dropped to fit the cap, 0 when the whole page fit.
    pub cut: usize,
}

const TLDS: &[&str] = &[
    "com", "org", "net", "ai", "io", "co", "us", "gov", "edu", "info", "app", "dev", "me", "biz",
    "uk", "ca", "de", "fr", "es", "it", "nl", "se", "no", "dk", "fi", "ch", "at", "be", "ie", "au", "nz",
    "in", "jp", "br", "mx", "eu", "tv", "xyz", "site", "online", "tech", "health", "news", "blog", "shop", "store",
];

/// The web addresses in a message: full ones ("https://containai.ai/"), and bare domains
/// ("containai.ai", "docs.example.org/page") whose last label is a known ending, so that
/// "e.g.", "Node.js" and "file.txt" are left alone. At most four, in order, no repeats.
pub fn find_urls(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in text.split(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | '"' | '\'' | '(' | ')' | '[' | ']' | ',' | ';')) {
        let token = raw.trim_end_matches(|c: char| matches!(c, '.' | '!' | '?' | ':'));
        if token.is_empty() || token.contains('@') { continue; }
        let lower = token.to_lowercase();
        let url = if lower.starts_with("http://") || lower.starts_with("https://") {
            if token.len() < 11 { continue; }
            token.to_string()
        } else {
            let host = lower.split('/').next().unwrap_or("");
            let labels: Vec<&str> = host.split('.').collect();
            if labels.len() < 2 || labels.iter().any(|l| l.is_empty() || !l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')) { continue; }
            let last = labels[labels.len() - 1];
            if !TLDS.contains(&last) { continue; }
            // "something.co" with no path is more often a typo than a site; "x.ai" and
            // "containai.ai" are real. Two-letter endings need a path or a longer name.
            if last.len() == 2 && labels.len() == 2 && labels[0].len() < 3 && !lower.contains('/') { continue; }
            format!("https://{}", token)
        };
        if !out.iter().any(|u| u.eq_ignore_ascii_case(&url)) { out.push(url); }
        if out.len() == 4 { break; }
    }
    out
}

/// Fetch one page and keep its readable text, at most `cap` characters. PDFs go through
/// the same extractor the Knowledge Base uses. Ten seconds, then it is reported as slow.
pub async fn read_page(url: &str, cap: usize) -> Result<ReadPage, String> {
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0")
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client.get(url).send().await.map_err(|e| {
        if e.is_timeout() { "the site did not answer within ten seconds".to_string() } else { format!("could not reach the site: {}", e) }
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("the site answered {}", status));
    }
    let content_type = response.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    let bytes = response.bytes().await.map_err(|e| format!("could not read the page: {}", e))?;
    let (title, text) = if content_type.contains("pdf") || bytes.starts_with(b"%PDF") {
        let text = pdf_extract::extract_text_from_mem(&bytes).map_err(|e| format!("could not read the PDF: {}", e))?;
        (String::new(), text)
    } else {
        let html = String::from_utf8_lossy(&bytes).into_owned();
        (html_title(&html), crate::web_search::extract_text_from_html(&html))
    };
    let text = tidy(&text);
    if text.trim().is_empty() {
        return Err("the page has no readable text (it may be built entirely in the browser)".to_string());
    }
    let total = text.chars().count();
    let kept: String = text.chars().take(cap).collect();
    Ok(ReadPage { url: url.to_string(), title, text: kept, cut: total.saturating_sub(cap) })
}

fn html_title(html: &str) -> String {
    let lower = html.to_lowercase();
    let Some(start) = lower.find("<title") else { return String::new() };
    let Some(gt) = lower[start..].find('>') else { return String::new() };
    let after = &html[start + gt + 1..];
    let end = after.to_lowercase().find("</title>").unwrap_or(after.len());
    after[..end].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Collapse runs of spaces and blank lines.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank += 1;
            if blank == 1 { out.push('\n'); }
        } else {
            blank = 0;
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

/// The prompt block for pages that were read, and for the ones that could not be.
pub fn build_block(pages: &[ReadPage], failures: &[(String, String)]) -> String {
    let mut ctx = String::from("\n\n=== WEB PAGES THE USER POINTED AT ===\n");
    ctx.push_str("The user gave these web addresses. Each page was fetched just now, so this is its current text, not a memory or a search result. Answer about the page from this text.\n\n");
    for p in pages {
        ctx.push_str(&format!("📄 {} {}\n{}\n", p.url, if p.title.is_empty() { String::new() } else { format!("({})", p.title) }, p.text));
        if p.cut > 0 { ctx.push_str(&format!("[... {} more characters not shown]\n", p.cut)); }
        ctx.push('\n');
    }
    for (url, why) in failures {
        ctx.push_str(&format!("⚠️ {} could not be read: {}. Tell the user plainly; do not guess at what the page says.\n\n", url, why));
    }
    ctx.push_str("=== END OF WEB PAGES ===\n\n");
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addresses_are_found_and_lookalikes_are_not() {
        assert_eq!(find_urls("I own this site, check it: https://containai.ai/"), vec!["https://containai.ai/"]);
        assert_eq!(find_urls("just go to containai.ai, can you scan that page?"), vec!["https://containai.ai"]);
        assert_eq!(find_urls("see docs.example.org/guide and www.hfcuvt.com/business."), vec!["https://docs.example.org/guide", "https://www.hfcuvt.com/business"]);
        assert!(find_urls("e.g. the file notes.txt uses Node.js and Python 3.12").is_empty());
        assert!(find_urls("write to matt@containai.ai about it").is_empty());
        assert!(find_urls("the ratio is 1.5 and it.co is odd").is_empty());
        assert_eq!(find_urls("https://a.org/x https://a.org/x https://b.org https://c.org https://d.org https://e.org").len(), 4);
    }

    #[test]
    fn the_title_and_text_are_tidied() {
        assert_eq!(html_title("<html><head><title>\n  Contain  AI\n</title></head></html>"), "Contain AI");
        assert_eq!(html_title("<html><body>no title</body></html>"), "");
        assert_eq!(tidy("  a   b  \n\n\n\n c \n"), "a b\n\nc");
    }

    #[test]
    fn the_block_names_what_was_read_and_what_was_not() {
        let page = ReadPage { url: "https://containai.ai".into(), title: "Contain AI".into(), text: "Zynkbot is a private assistant.".into(), cut: 120 };
        let block = build_block(&[page], &[("https://down.example".into(), "the site answered 503".into())]);
        assert!(block.contains("📄 https://containai.ai (Contain AI)\nZynkbot is a private assistant."));
        assert!(block.contains("120 more characters not shown"));
        assert!(block.contains("https://down.example could not be read: the site answered 503"));
    }
}
