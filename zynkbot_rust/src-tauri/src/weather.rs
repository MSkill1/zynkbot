//! Weather without a model call (2026-10-25, Matt): a weather question is answered from
//! Open-Meteo (no key, no account; free for non-commercial use; "Weather data by
//! Open-Meteo" is the credit its licence asks for) before any model is involved. The
//! place comes from the question ("weather in Boston tomorrow"), else from a remembered
//! home town (a memory saying where the user lives or their mailing address); with
//! neither, the reply asks for the city. Any failure (offline, unknown place) falls
//! through to the ordinary model path, which may offer a web search.

use regex::Regex;

#[derive(Debug, Clone, PartialEq)]
pub enum When { Now, Tomorrow, Week }

#[derive(Debug, Clone, PartialEq)]
pub struct WeatherAsk { pub place: Option<String>, pub when: When }

const STOP_TAIL: &[&str] = &["today", "tonight", "tomorrow", "this", "right", "now", "week", "weekend", "like", "outside", "please", "be", "the", "next", "few", "days", "morning", "afternoon", "evening", "later"];

/// Is this a weather question, and for where and when? None for anything else,
/// including questions about what was said about the weather.
pub fn detect(query: &str) -> Option<WeatherAsk> {
    let t = query.to_lowercase();
    let t = t.trim();
    let triggers = ["weather", "forecast", "temperature outside", "temperature today", "temperature tomorrow", "how hot is it", "how cold is it", "how warm is it", "going to rain", "will it rain", "is it raining", "chance of rain", "going to snow", "will it snow", "is it snowing", "how windy", "humidity today"];
    if !triggers.iter().any(|w| t.contains(w)) { return None; }
    // Memory questions about the weather are not forecasts.
    if ["did i", "i said", "remember", "memory", "memories", "yesterday", "last week", "what was the weather"].iter().any(|w| t.contains(w)) { return None; }
    let when = if t.contains("tomorrow") { When::Tomorrow }
        else if ["this week", "the week", "weekend", "next few days", "next days", "7 day", "seven day", "forecast"].iter().any(|w| t.contains(w)) && !t.contains("today") { When::Week }
        else { When::Now };
    let place = place_in(query);
    Some(WeatherAsk { place, when })
}

fn clean_place(raw: &str) -> Option<String> {
    let mut words: Vec<&str> = raw.split_whitespace().map(|w| w.trim_matches(|c: char| matches!(c, '?' | '!' | '.' | ',' | '"' | '\''))).filter(|w| !w.is_empty()).collect();
    while let Some(last) = words.last() { if STOP_TAIL.contains(&last.to_lowercase().as_str()) { words.pop(); } else { break; } }
    if words.is_empty() { return None; }
    let joined = words.join(" ");
    // keep a comma between city and state ("Rutland, Vermont")
    let joined = raw.find(',').map(|_| {
        let parts: Vec<String> = raw.split(',').map(|p| p.split_whitespace().map(|w| w.trim_matches(|c: char| matches!(c, '?' | '!' | '.' | '"' | '\''))).filter(|w| !w.is_empty() && !STOP_TAIL.contains(&w.to_lowercase().as_str())).collect::<Vec<_>>().join(" ")).filter(|p| !p.is_empty()).collect();
        parts.join(", ")
    }).filter(|s| !s.is_empty()).unwrap_or(joined);
    if joined.to_lowercase() == "here" || joined.to_lowercase() == "my area" { return None; }
    Some(joined)
}

/// "in Boston", "for Rutland, Vermont tomorrow", "Boston weather".
fn place_in(query: &str) -> Option<String> {
    let re = Regex::new(r"(?i)\b(?:in|for|at|around|near)\s+([A-Za-z][A-Za-z.'-]*(?:,?\s+[A-Za-z][A-Za-z.'-]*){0,3})").ok()?;
    if let Some(c) = re.captures(query) {
        if let Some(p) = clean_place(c.get(1).map(|m| m.as_str()).unwrap_or("")) {
            if !["the", "a", "my", "our"].contains(&p.to_lowercase().as_str()) { return Some(p); }
        }
    }
    let re2 = Regex::new(r"(?:^|\s)([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:weather|forecast)\b").ok()?;
    if let Some(c) = re2.captures(query) {
        let p = c.get(1)?.as_str();
        if !["What", "Whats", "The", "Todays", "Tomorrows", "Current"].contains(&p) { return clean_place(p); }
    }
    None
}

/// The user's home from their memories: a mailing address ("…, Rutland, Vermont, …") or
/// "lives in X". Most recent first. None when nothing says.
pub async fn remembered_place(user_id: &str) -> Option<String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await.ok()?;
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT content FROM memories WHERE user_id = ? AND (temporal_status IS NULL OR temporal_status != 'superseded') \
         AND (expires_at IS NULL OR substr(expires_at, 1, 10) >= date('now')) \
         AND (content LIKE '%lives in %' OR content LIKE '%live in %' OR content LIKE '%mailing address%' OR content LIKE '%home is in %' OR content LIKE '%moved to %') \
         ORDER BY created_at DESC LIMIT 30")
        .bind(user_id).fetch_all(&pool).await.ok()?;
    let addr = Regex::new(r"(?i)mailing address[^,]*,[^,]*,\s*([A-Z][A-Za-z.'-]+(?:\s[A-Z][A-Za-z.'-]+)?),\s*([A-Z][A-Za-z.'-]+(?:\s[A-Z][A-Za-z.'-]+)?)").ok()?;
    let lives = Regex::new(r"(?i)\b(?:lives? in|home is in|moved to)\s+([A-Z][A-Za-z.'-]+(?:\s[A-Z][A-Za-z.'-]+)?(?:,\s*[A-Z][A-Za-z.'-]+(?:\s[A-Z][A-Za-z.'-]+)?)?)").ok()?;
    for (c,) in &rows {
        if let Some(m) = addr.captures(c) { return Some(format!("{}, {}", &m[1], &m[2])); }
    }
    for (c,) in &rows {
        if let Some(m) = lives.captures(c) { return clean_place(&m[1]); }
    }
    None
}

fn describe(code: i64) -> &'static str {
    match code {
        0 => "clear", 1 => "mostly clear", 2 => "partly cloudy", 3 => "overcast",
        45 | 48 => "fog", 51 | 53 | 55 => "drizzle", 56 | 57 => "freezing drizzle",
        61 => "light rain", 63 => "rain", 65 => "heavy rain", 66 | 67 => "freezing rain",
        71 => "light snow", 73 => "snow", 75 => "heavy snow", 77 => "snow grains",
        80 => "light showers", 81 => "showers", 82 => "heavy showers", 85 => "snow showers", 86 => "heavy snow showers",
        95 => "thunderstorms", 96 | 99 => "thunderstorms with hail", _ => "mixed conditions",
    }
}

struct Geo { name: String, region: String, lat: f64, lon: f64 }

async fn geocode(place: &str) -> Result<Geo, String> {
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build().map_err(|e| e.to_string())?;
    let v: serde_json::Value = client.get("https://geocoding-api.open-meteo.com/v1/search")
        .query(&[("name", place), ("count", "1"), ("language", "en"), ("format", "json")])
        .send().await.map_err(|e| format!("geocoding: {}", e))?
        .json().await.map_err(|e| format!("geocoding: {}", e))?;
    let r = v.get("results").and_then(|r| r.as_array()).and_then(|a| a.first()).ok_or_else(|| format!("no place called {}", place))?;
    let region = [r.get("admin1"), r.get("country")].iter().filter_map(|x| x.and_then(|y| y.as_str())).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(", ");
    Ok(Geo {
        name: r.get("name").and_then(|n| n.as_str()).unwrap_or(place).to_string(),
        region,
        lat: r.get("latitude").and_then(|n| n.as_f64()).ok_or("geocoding: no latitude")?,
        lon: r.get("longitude").and_then(|n| n.as_f64()).ok_or("geocoding: no longitude")?,
    })
}

/// The answer text, or Err when it cannot be had (offline, unknown place).
pub async fn answer(ask: &WeatherAsk, user_id: &str) -> Result<String, String> {
    let place = match &ask.place {
        Some(p) => p.clone(),
        None => match remembered_place(user_id).await {
            Some(p) => p,
            None => return Ok("Which city? Tell me the place and I'll look up the weather.".to_string()),
        },
    };
    let geo = geocode(&place).await?;
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(8)).build().map_err(|e| e.to_string())?;
    let v: serde_json::Value = client.get("https://api.open-meteo.com/v1/forecast")
        .query(&[
            ("latitude", geo.lat.to_string()), ("longitude", geo.lon.to_string()),
            ("current", "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m".to_string()),
            ("daily", "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max".to_string()),
            ("temperature_unit", "fahrenheit".to_string()), ("wind_speed_unit", "mph".to_string()),
            ("timezone", "auto".to_string()), ("forecast_days", "7".to_string()),
        ])
        .send().await.map_err(|e| format!("forecast: {}", e))?
        .json().await.map_err(|e| format!("forecast: {}", e))?;
    let cur = v.get("current").ok_or("forecast: no current block")?;
    let daily = v.get("daily").ok_or("forecast: no daily block")?;
    let f = |x: &serde_json::Value, k: &str| x.get(k).and_then(|n| n.as_f64());
    let arr = |k: &str| daily.get(k).and_then(|a| a.as_array()).cloned().unwrap_or_default();
    let dates = arr("time"); let codes = arr("weather_code"); let highs = arr("temperature_2m_max"); let lows = arr("temperature_2m_min"); let pops = arr("precipitation_probability_max");
    let day_line = |i: usize, label: &str| -> Option<String> {
        let code = codes.get(i)?.as_i64()?;
        let hi = highs.get(i)?.as_f64()?; let lo = lows.get(i)?.as_f64()?;
        let pop = pops.get(i).and_then(|p| p.as_f64()).unwrap_or(0.0);
        let rain = if pop >= 20.0 { format!(", {:.0}% chance of rain", pop) } else { String::new() };
        Some(format!("{}: {}, high {:.0}°F, low {:.0}°F{}.", label, describe(code), hi, lo, rain))
    };
    let weekday = |i: usize| -> String {
        dates.get(i).and_then(|d| d.as_str()).and_then(|d| chrono::NaiveDate::parse_from_str(d, "%Y-%m-%d").ok()).map(|d| d.format("%A").to_string()).unwrap_or_else(|| format!("Day {}", i + 1))
    };
    let where_ = if geo.region.is_empty() { geo.name.clone() } else { format!("{}, {}", geo.name, geo.region) };
    let mut out = String::new();
    match ask.when {
        When::Now => {
            if let (Some(t), Some(code)) = (f(cur, "temperature_2m"), cur.get("weather_code").and_then(|c| c.as_i64())) {
                out.push_str(&format!("{} right now: {:.0}°F and {}", where_, t, describe(code)));
                if let Some(a) = f(cur, "apparent_temperature") { if (a - t).abs() >= 3.0 { out.push_str(&format!(", feels like {:.0}°F", a)); } }
                if let Some(w) = f(cur, "wind_speed_10m") { out.push_str(&format!(", wind {:.0} mph", w)); }
                out.push_str(". ");
            }
            if let Some(l) = day_line(0, "Today") { out.push_str(&l); out.push(' '); }
            if let Some(l) = day_line(1, "Tomorrow") { out.push_str(&l); }
        }
        When::Tomorrow => {
            out.push_str(&format!("{}, ", where_));
            if let Some(l) = day_line(1, &format!("tomorrow ({})", weekday(1))) { out.push_str(&l); }
        }
        When::Week => {
            out.push_str(&format!("{}, the next seven days:\n", where_));
            for i in 0..dates.len().min(7) {
                let label = if i == 0 { "Today".to_string() } else if i == 1 { "Tomorrow".to_string() } else { weekday(i) };
                if let Some(l) = day_line(i, &label) { out.push_str("- "); out.push_str(&l); out.push('\n'); }
            }
        }
    }
    if out.trim().is_empty() { return Err("forecast: empty".into()); }
    out = out.trim_end().to_string();
    out.push_str("\n\nWeather data by Open-Meteo.");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weather_questions_are_recognised_with_place_and_time() {
        assert_eq!(detect("What's the weather in Boston tomorrow?"), Some(WeatherAsk { place: Some("Boston".into()), when: When::Tomorrow }));
        assert_eq!(detect("How cold is it outside right now?"), Some(WeatherAsk { place: None, when: When::Now }));
        assert_eq!(detect("Rutland weather this week"), Some(WeatherAsk { place: Some("Rutland".into()), when: When::Week }));
        assert_eq!(detect("weather for Rutland, Vermont today"), Some(WeatherAsk { place: Some("Rutland, Vermont".into()), when: When::Now }));
        assert_eq!(detect("Is it going to rain in Denver this weekend?"), Some(WeatherAsk { place: Some("Denver".into()), when: When::Week }));
    }
    #[test]
    fn not_weather_questions() {
        assert_eq!(detect("What did I say about the weather yesterday?"), None);
        assert_eq!(detect("Does it snow in Vermont?"), None);
        assert_eq!(detect("Set a timer for ten minutes"), None);
        assert_eq!(detect("Remember: the weather station is on the roof"), None);
    }
}
