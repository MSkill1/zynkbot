/// Knowledge Base - External reference document system
///
/// Allows users to point Zynkbot at a directory of reference materials
/// (PDFs, text files, markdown, etc.) that can be searched and referenced
/// without polluting the personal memory system.
///
/// Use cases:
/// - Journalist: Country guides, maps, safety protocols
/// - Researcher: Papers, datasets, documentation
/// - Student: Textbooks, course materials
/// - Developer: API docs, code examples
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseFile {
    pub path: String,
    pub name: String,
    pub size: u64,
    pub modified: String,
    pub file_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseSearchResult {
    pub file_path: String,
    pub file_name: String,
    pub excerpt: String,
    pub relevance_score: f32,
}

/// Supported file types for knowledge base
const SUPPORTED_EXTENSIONS: &[&str] = &[
    "txt", "md", "json", "csv", "log",
    "rs", "js", "jsx", "ts", "tsx", "py", "java", "cpp", "c", "h",
    "html", "css", "xml", "yaml", "yml", "toml",
];

/// Check if file extension is supported
fn is_supported_file(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return SUPPORTED_EXTENSIONS.contains(&ext_str.to_lowercase().as_str());
        }
    }
    false
}

/// Scan directory and return list of all supported files
pub fn scan_knowledge_base_directory(directory: &str) -> Result<Vec<KnowledgeBaseFile>, String> {
    let path = Path::new(directory);

    if !path.exists() {
        return Err(format!("Directory does not exist: {}", directory));
    }

    if !path.is_dir() {
        return Err(format!("Path is not a directory: {}", directory));
    }

    let mut files = Vec::new();

    for entry in WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let entry_path = entry.path();

        // Skip directories and unsupported files
        if !entry_path.is_file() || !is_supported_file(entry_path) {
            continue;
        }

        // Get file metadata
        if let Ok(metadata) = fs::metadata(entry_path) {
            let file_name = entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let file_type = entry_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_string();

            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            files.push(KnowledgeBaseFile {
                path: entry_path.to_string_lossy().to_string(),
                name: file_name,
                size: metadata.len(),
                modified,
                file_type,
            });
        }
    }

    Ok(files)
}

/// Read file content (handles text files, will expand to PDFs later)
pub fn read_knowledge_base_file(file_path: &str) -> Result<String, String> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Err(format!("File does not exist: {}", file_path));
    }

    if !path.is_file() {
        return Err(format!("Path is not a file: {}", file_path));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    Ok(content)
}

/// Simple keyword-based search in knowledge base files
/// Returns files containing the search terms with relevant excerpts
pub fn search_knowledge_base(
    directory: &str,
    query: &str,
) -> Result<Vec<KnowledgeBaseSearchResult>, String> {
    let files = scan_knowledge_base_directory(directory)?;
    let mut results = Vec::new();

    let query_lower = query.to_lowercase();
    let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

    for file in files {
        // Try to read file content
        if let Ok(content) = read_knowledge_base_file(&file.path) {
            let content_lower = content.to_lowercase();

            // Count how many query terms appear in the file
            let mut matches = 0;
            for term in &query_terms {
                if content_lower.contains(term) {
                    matches += 1;
                }
            }

            // If file contains at least one query term, include it
            if matches > 0 {
                // Calculate simple relevance score (percentage of terms matched)
                let relevance = matches as f32 / query_terms.len() as f32;

                // Extract excerpt containing first match
                let excerpt = extract_excerpt(&content, &query_terms);

                results.push(KnowledgeBaseSearchResult {
                    file_path: file.path.clone(),
                    file_name: file.name.clone(),
                    excerpt,
                    relevance_score: relevance,
                });
            }
        }
    }

    // Sort by relevance (highest first)
    results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Extract a relevant excerpt from content containing query terms
fn extract_excerpt(content: &str, query_terms: &[&str]) -> String {
    let content_lower = content.to_lowercase();

    // Find the first occurrence of any query term
    let mut earliest_pos = None;
    for term in query_terms {
        if let Some(pos) = content_lower.find(term) {
            if earliest_pos.map_or(true, |e| pos < e) {
                earliest_pos = Some(pos);
            }
        }
    }

    if let Some(pos) = earliest_pos {
        // Extract ~200 chars before and after the match
        let start = pos.saturating_sub(200);
        let end = (pos + 200).min(content.len());

        let excerpt = &content[start..end];

        // Clean up the excerpt
        let excerpt = excerpt.trim();

        // Add ellipsis if truncated
        let prefix = if start > 0 { "..." } else { "" };
        let suffix = if end < content.len() { "..." } else { "" };

        format!("{}{}{}", prefix, excerpt, suffix)
    } else {
        // Fallback: return first 200 chars
        let excerpt = &content[0..200.min(content.len())];
        format!("{}...", excerpt.trim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_file() {
        assert!(is_supported_file(Path::new("test.txt")));
        assert!(is_supported_file(Path::new("test.md")));
        assert!(is_supported_file(Path::new("test.pdf")));
        assert!(is_supported_file(Path::new("test.rs")));
        assert!(!is_supported_file(Path::new("test.exe")));
        assert!(!is_supported_file(Path::new("test.dll")));
    }

    #[test]
    fn test_extract_excerpt() {
        let content = "This is a long document about Afghanistan. The country has rich cultural customs around photography and hospitality.";
        let terms = vec!["afghanistan", "customs"];
        let excerpt = extract_excerpt(content, &terms);
        assert!(excerpt.contains("Afghanistan"));
    }
}
