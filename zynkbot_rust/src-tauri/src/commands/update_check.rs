//! Once-a-day version check (Matt, 2026-11-26). The page calls this at most once a
//! day when the switch is on and the containment mode is not HIPAA. It fetches one
//! small static file from containai.ai that names the newest release. The request
//! carries the running version and the platform (in the query and the user agent) and
//! nothing else: no identifier, no account, nothing from the user's data. The server's
//! ordinary access log then shows how many copies of each version are in use, which is
//! the only thing learned. Disclosed in docs/FEATURES.md and on containai.ai/zynkbot/privacy/.

use std::time::Duration;

const LATEST_URL: &str = "https://containai.ai/zynkbot/latest.json";
const RELEASES_PAGE: &str = "https://containai.ai/zynkbot/";

#[derive(serde::Serialize)]
pub struct UpdateCheck {
    /// The version this build runs.
    pub current: String,
    /// The newest release named by the site, if the file could be read.
    pub latest: Option<String>,
    /// True when `latest` is newer than `current`.
    pub newer: bool,
    /// Where to get it.
    pub url: String,
}

/// Numeric compare of dotted versions; a release outranks its own pre-release
/// ("0.9.7" is newer than "0.9.7-beta2"), and anything else after the numbers is ignored.
fn is_newer(latest: &str, current: &str) -> bool {
    fn parts(v: &str) -> (Vec<u64>, bool) {
        let v = v.trim().trim_start_matches('v');
        let (nums, pre) = match v.find('-') {
            Some(i) => (&v[..i], true),
            None => (v, false),
        };
        (nums.split('.').map(|p| p.parse::<u64>().unwrap_or(0)).collect(), pre)
    }
    let (l, l_pre) = parts(latest);
    let (c, c_pre) = parts(current);
    let n = l.len().max(c.len());
    for i in 0..n {
        let a = l.get(i).copied().unwrap_or(0);
        let b = c.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    c_pre && !l_pre
}

/// The last answer, kept in `<app data dir>/update_check.json` so the page can ask on
/// every start and the site is still read at most once a day. A result for another
/// version (the app was updated since) does not count.
#[derive(serde::Serialize, serde::Deserialize)]
struct Cached {
    checked_at: u64,
    current: String,
    latest: Option<String>,
    url: String,
}

const DAY_SECS: u64 = 24 * 60 * 60;

fn cache_path() -> std::path::PathBuf {
    crate::db::get_app_data_dir().join("update_check.json")
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn read_cache(current: &str) -> Option<Cached> {
    let text = std::fs::read_to_string(cache_path()).ok()?;
    let c: Cached = serde_json::from_str(&text).ok()?;
    if c.current != current || now_secs().saturating_sub(c.checked_at) >= DAY_SECS {
        return None;
    }
    Some(c)
}

fn write_cache(c: &Cached) {
    if let Ok(text) = serde_json::to_string(c) {
        let _ = std::fs::write(cache_path(), text);
    }
}

#[tauri::command]
pub async fn check_for_update(app: tauri::AppHandle) -> Result<UpdateCheck, String> {
    let current = app.package_info().version.to_string();
    let platform = std::env::consts::OS;
    if let Some(c) = read_cache(&current) {
        let newer = c.latest.as_deref().map(|l| is_newer(l, &current)).unwrap_or(false);
        return Ok(UpdateCheck { current, latest: c.latest, newer, url: c.url });
    }
    let url = format!("{LATEST_URL}?v={current}&p={platform}");
    let client = reqwest::Client::builder()
        .user_agent(format!("Zynkbot/{current} ({platform})"))
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let body: serde_json::Value = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("version check: {e}"))?
        .error_for_status()
        .map_err(|e| format!("version check: {e}"))?
        .json()
        .await
        .map_err(|e| format!("version check: {e}"))?;
    let latest = body.get("version").and_then(|v| v.as_str()).map(|s| s.to_string());
    let page = body.get("url").and_then(|v| v.as_str()).unwrap_or(RELEASES_PAGE).to_string();
    let newer = latest.as_deref().map(|l| is_newer(l, &current)).unwrap_or(false);
    log::info!("Version check: running {current} on {platform}, newest {}{}", latest.as_deref().unwrap_or("?"), if newer { " (newer)" } else { "" });
    write_cache(&Cached { checked_at: now_secs(), current: current.clone(), latest: latest.clone(), url: page.clone() });
    Ok(UpdateCheck { current, latest, newer, url: page })
}

#[cfg(test)]
mod tests {
    use super::is_newer;

    #[test]
    fn version_order() {
        assert!(is_newer("0.9.7", "0.9.6"));
        assert!(is_newer("1.0.0", "0.9.6"));
        assert!(!is_newer("0.9.6", "0.9.6"));
        assert!(!is_newer("0.9.5", "0.9.6"));
        assert!(is_newer("0.9.7", "0.9.7-beta2"));
        assert!(!is_newer("0.9.7-beta2", "0.9.7"));
        assert!(is_newer("v0.10.0", "0.9.6"));
    }
}
