mod ollama;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ollama::OllamaClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{info, warn};

const CHARTS_URL: &str = "https://raw.githubusercontent.com/mhollingshead/billboard-hot-100/main/all.json";
const USER_AGENT: &str = "LyrIQ/0.1.0 (local lyric analyzer)";
const ANALYSIS_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "themes": { "type": "array", "items": { "type": "string" } },
    "mood": { "type": "string" },
    "summary": { "type": "string" },
    "explicit": { "type": "boolean" }
  },
  "required": ["themes", "mood", "summary", "explicit"]
}"#;

#[derive(Parser)]
#[command(name = "lyriq", about = "Analyze Billboard Hot 100 lyrics with a local LLM")]
struct Cli {
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download Billboard Hot 100 history and write unique songs
    Charts {
        #[arg(long, default_value_t = false)]
        refresh: bool,
    },
    /// Fetch lyrics from LRCLIB into data/lyrics
    Lyrics {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 500)]
        delay_ms: u64,
    },
    /// Analyze saved lyrics with Ollama
    Analyze {
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        peak_max: Option<u32>,
        #[arg(long, env = "OLLAMA_HOST", default_value = "http://127.0.0.1:11434")]
        ollama_host: String,
        #[arg(long, env = "OLLAMA_MODEL", default_value = "qwen3:14b")]
        ollama_model: String,
    },
}

#[derive(Debug, Deserialize)]
struct Chart {
    data: Vec<ChartSong>,
}

#[derive(Debug, Deserialize)]
struct ChartSong {
    song: String,
    artist: String,
    peak_position: u32,
    weeks_on_chart: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Song {
    artist: String,
    title: String,
    peak: u32,
    weeks: u32,
}

#[derive(Debug, Deserialize)]
struct LrcHit {
    #[serde(rename = "plainLyrics")]
    plain_lyrics: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Analysis {
    themes: Vec<String>,
    mood: String,
    summary: String,
    explicit: bool,
}

#[derive(Debug, Serialize)]
struct AnalysisFile<'a> {
    artist: &'a str,
    title: &'a str,
    peak: u32,
    weeks: u32,
    themes: Vec<String>,
    mood: String,
    summary: String,
    explicit: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Command::Charts { refresh } => charts(&cli.data_dir, refresh).await,
        Command::Lyrics { limit, delay_ms } => lyrics(&cli.data_dir, limit, delay_ms).await,
        Command::Analyze {
            limit,
            peak_max,
            ollama_host,
            ollama_model,
        } => analyze(&cli.data_dir, limit, peak_max, &ollama_host, &ollama_model).await,
    }
}

async fn charts(data_dir: &Path, refresh: bool) -> Result<()> {
    let charts_dir = data_dir.join("charts");
    tokio::fs::create_dir_all(&charts_dir).await?;
    let dump = charts_dir.join("hot-100.json");
    if refresh || !dump.exists() {
        info!("downloading Billboard Hot 100 history");
        let http = http_client()?;
        let bytes = http.get(CHARTS_URL).send().await?.error_for_status()?.bytes().await?;
        tokio::fs::write(&dump, &bytes).await?;
    }
    let raw = tokio::fs::read(&dump).await?;
    let weeks: Vec<Chart> = serde_json::from_slice(&raw).context("hot-100.json was not Billboard chart JSON")?;
    let songs = unique_songs(&weeks);
    let songs_path = data_dir.join("songs.json");
    tokio::fs::write(&songs_path, serde_json::to_vec(&songs)?).await?;
    info!("wrote {} unique songs to {}", songs.len(), songs_path.display());
    Ok(())
}

async fn lyrics(data_dir: &Path, limit: Option<usize>, delay_ms: u64) -> Result<()> {
    let songs = load_songs(data_dir).await?;
    let lyrics_dir = data_dir.join("lyrics");
    tokio::fs::create_dir_all(&lyrics_dir).await?;
    let http = http_client()?;
    let delay = Duration::from_millis(delay_ms);
    let mut saved = 0usize;
    let mut skipped = 0usize;
    let mut missing = 0usize;
    for song in &songs {
        if limit.is_some_and(|n| saved + missing >= n) {
            break;
        }
        let path = lyrics_dir.join(format!("{}.txt", slug(&song.artist, &song.title)));
        if path.exists() {
            if let Ok(text) = tokio::fs::read_to_string(&path).await {
                if !lyric_body(&text).is_empty() {
                    skipped += 1;
                    continue;
                }
            }
        }
        match fetch_lyrics(&http, song, delay).await {
            Ok(Some(body)) => {
                tokio::fs::write(&path, format_lyrics(song, &body)).await?;
                saved += 1;
            }
            Ok(None) => {
                tokio::fs::write(&path, format_lyrics(song, "")).await?;
                missing += 1;
            }
            Err(error) => {
                warn!("{} — {}: {error}", song.artist, song.title);
                missing += 1;
            }
        }
        if (saved + missing).is_multiple_of(100) {
            info!("lyrics progress: {saved} saved, {missing} missing, {skipped} skipped");
        }
        tokio::time::sleep(delay).await;
    }
    info!("lyrics done: {saved} saved, {missing} missing, {skipped} skipped");
    Ok(())
}

async fn analyze(
    data_dir: &Path,
    limit: Option<usize>,
    peak_max: Option<u32>,
    ollama_host: &str,
    ollama_model: &str,
) -> Result<()> {
    let songs = load_songs(data_dir).await?;
    let lyrics_dir = data_dir.join("lyrics");
    let analysis_dir = data_dir.join("analysis");
    tokio::fs::create_dir_all(&analysis_dir).await?;
    let client = OllamaClient::new(ollama_host, ollama_model)?;
    let mut done = 0usize;
    let mut skipped = 0usize;
    for song in &songs {
        if peak_max.is_some_and(|max| song.peak > max) {
            continue;
        }
        if limit.is_some_and(|n| done >= n) {
            break;
        }
        let stem = slug(&song.artist, &song.title);
        let lyrics_path = lyrics_dir.join(format!("{stem}.txt"));
        let analysis_path = analysis_dir.join(format!("{stem}.json"));
        if analysis_path.exists() {
            skipped += 1;
            continue;
        }
        let Ok(text) = tokio::fs::read_to_string(&lyrics_path).await else {
            skipped += 1;
            continue;
        };
        let body = lyric_body(&text);
        if body.is_empty() {
            skipped += 1;
            continue;
        }
        let prompt = format!(
            "Analyze these song lyrics. Return JSON only.\n\nArtist: {}\nTitle: {}\n\nLyrics:\n{body}",
            song.artist, song.title
        );
        match client.complete::<Analysis>(&prompt, ANALYSIS_SCHEMA).await {
            Ok(analysis) => {
                let file = AnalysisFile {
                    artist: &song.artist,
                    title: &song.title,
                    peak: song.peak,
                    weeks: song.weeks,
                    themes: analysis.themes,
                    mood: analysis.mood,
                    summary: analysis.summary,
                    explicit: analysis.explicit,
                };
                tokio::fs::write(&analysis_path, serde_json::to_vec_pretty(&file)?).await?;
                done += 1;
            }
            Err(error) => warn!("{} — {}: {error}", song.artist, song.title),
        }
    }
    info!("analyze done: {done} written, {skipped} skipped");
    Ok(())
}

async fn load_songs(data_dir: &Path) -> Result<Vec<Song>> {
    let path = data_dir.join("songs.json");
    let raw = tokio::fs::read(&path)
        .await
        .with_context(|| format!("run `lyriq charts` first ({})", path.display()))?;
    serde_json::from_slice(&raw).context("songs.json was not valid")
}

fn http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(120))
        .build()?)
}

fn unique_songs(charts: &[Chart]) -> Vec<Song> {
    let mut map: HashMap<(String, String), Song> = HashMap::new();
    for chart in charts {
        for entry in &chart.data {
            let key = (entry.artist.to_lowercase(), entry.song.to_lowercase());
            map.entry(key)
                .and_modify(|song| {
                    song.peak = song.peak.min(entry.peak_position);
                    song.weeks = song.weeks.max(entry.weeks_on_chart);
                })
                .or_insert(Song {
                    artist: entry.artist.clone(),
                    title: entry.song.clone(),
                    peak: entry.peak_position,
                    weeks: entry.weeks_on_chart,
                });
        }
    }
    let mut songs: Vec<_> = map.into_values().collect();
    songs.sort_by(|a, b| {
        a.peak
            .cmp(&b.peak)
            .then(b.weeks.cmp(&a.weeks))
            .then(a.artist.cmp(&b.artist))
            .then(a.title.cmp(&b.title))
    });
    songs
}

// ponytail: collisions if titles differ only by / vs _; hash suffix if that shows up
fn slug(artist: &str, title: &str) -> String {
    let mut name = format!("{}__{}", sanitize(artist), sanitize(title));
    if name.len() > 200 {
        name.truncate(200);
    }
    name
}

fn sanitize(value: &str) -> String {
    value.replace(['/', '\\', '\0'], "_")
}

fn format_lyrics(song: &Song, body: &str) -> String {
    format!(
        "# artist: {}\n# title: {}\n# peak: {}\n# weeks: {}\n\n{body}",
        song.artist, song.title, song.peak, song.weeks
    )
}

fn lyric_body(text: &str) -> &str {
    text.split_once("\n\n")
        .map(|(_, body)| body.trim())
        .unwrap_or("")
        .trim()
}

fn first_cut(lower: &str, markers: &[&str]) -> Option<usize> {
    markers.iter().filter_map(|marker| lower.find(marker)).min()
}

fn query_artist(artist: &str) -> &str {
    let lower = artist.to_ascii_lowercase();
    let cut = first_cut(
        &lower,
        &[
            " (featuring ",
            " (feat. ",
            " (feat ",
            " (with ",
            " featuring ",
            " feat. ",
            " feat ",
            " a duet with ",
            " starring ",
            " and his orchestra",
            " and her orchestra",
            " and the ",
            " and wings",
            " with ",
            " x ",
        ],
    )
    .map_or(artist.trim(), |index| artist[..index].trim());
    // ponytail: first comma/"/" — breaks "Earth, Wind & Fire" style names if those are still empty
    let cut = cut.split_once(" & ").map_or(cut, |(head, _)| head.trim());
    let cut = cut.split_once(", ").map_or(cut, |(head, _)| head.trim());
    cut.split_once('/').map_or(cut, |(head, _)| head.trim())
}

fn query_title(title: &str) -> String {
    let lower = title.to_ascii_lowercase();
    let cut = first_cut(&lower, &[" (from ", " (live "]).map_or(title, |index| &title[..index]);
    let cut = cut.split_once(" - ").map_or(cut, |(head, _)| head);
    let cut = cut.split_once('/').map_or(cut, |(head, _)| head).trim();
    cut.trim_end_matches(['!', '.']).trim().to_string()
}

fn pick_lyrics(hits: &[LrcHit]) -> Option<String> {
    hits.iter().find_map(|hit| {
        hit.plain_lyrics
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToString::to_string)
    })
}

async fn get_ok(builder: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        let response = builder.try_clone().context("request was not cloneable")?.send().await?;
        let status = response.status();
        if attempts < 5
            && (status == reqwest::StatusCode::TOO_MANY_REQUESTS || status == reqwest::StatusCode::SERVICE_UNAVAILABLE)
        {
            let wait = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    2
                } else {
                    5
                })
                .max(1);
            tokio::time::sleep(Duration::from_secs(wait)).await;
            continue;
        }
        return Ok(response.error_for_status()?);
    }
}

async fn fetch_lrclib(http: &reqwest::Client, title: &str, artist: &str) -> Result<Option<String>> {
    let response = get_ok(
        http.get("https://lrclib.net/api/search")
            .query(&[("track_name", title), ("artist_name", artist)]),
    )
    .await?;
    let hits: Vec<LrcHit> = response.json().await?;
    Ok(pick_lyrics(&hits))
}

#[derive(Debug, Deserialize)]
struct NeteaseSearch {
    result: Option<NeteaseSearchResult>,
}

#[derive(Debug, Deserialize)]
struct NeteaseSearchResult {
    songs: Option<Vec<NeteaseSong>>,
}

#[derive(Debug, Deserialize)]
struct NeteaseSong {
    id: u64,
}

#[derive(Debug, Deserialize)]
struct NeteaseLyric {
    lrc: Option<NeteaseLrc>,
}

#[derive(Debug, Deserialize)]
struct NeteaseLrc {
    lyric: Option<String>,
}

fn plain_from_lrc(lrc: &str) -> Option<String> {
    let lines: Vec<&str> = lrc
        .lines()
        .map(|line| {
            let mut rest = line;
            while let Some(stripped) = rest.strip_prefix('[') {
                let Some((_, tail)) = stripped.split_once(']') else {
                    break;
                };
                rest = tail;
            }
            rest.trim()
        })
        .filter(|line| {
            !line.is_empty()
                && !line.starts_with("作词")
                && !line.starts_with("作曲")
                && !line.starts_with("编曲")
                && *line != "纯音乐"
        })
        .collect();
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

async fn fetch_netease(http: &reqwest::Client, title: &str, artist: &str, delay: Duration) -> Result<Option<String>> {
    let query = format!("{title} {artist}");
    let search: NeteaseSearch = get_ok(
        http.get("https://music.163.com/api/search/get")
            .header("Referer", "https://music.163.com/")
            .query(&[("type", "1"), ("s", query.as_str()), ("limit", "3")]),
    )
    .await?
    .json()
    .await?;
    let Some(id) = search
        .result
        .and_then(|result| result.songs)
        .and_then(|songs| songs.into_iter().next())
        .map(|song| song.id)
    else {
        return Ok(None);
    };
    tokio::time::sleep(delay).await;
    let id = id.to_string();
    let lyric: NeteaseLyric = get_ok(
        http.get("https://music.163.com/api/song/lyric")
            .header("Referer", "https://music.163.com/")
            .query(&[("id", id.as_str()), ("lv", "1"), ("kv", "1"), ("tv", "-1")]),
    )
    .await?
    .json()
    .await?;
    Ok(lyric.lrc.and_then(|lrc| lrc.lyric).as_deref().and_then(plain_from_lrc))
}

async fn fetch_lyrics(http: &reqwest::Client, song: &Song, delay: Duration) -> Result<Option<String>> {
    let artist = query_artist(&song.artist);
    let title = query_title(&song.title);
    if let Some(lyrics) = fetch_lrclib(http, &title, artist).await? {
        return Ok(Some(lyrics));
    }
    tokio::time::sleep(delay).await;
    fetch_netease(http, &title, artist, delay).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn unique_songs_keeps_best_peak_and_weeks() {
        let charts = vec![
            Chart {
                data: vec![ChartSong {
                    song: "Blinding Lights".into(),
                    artist: "The Weeknd".into(),
                    peak_position: 2,
                    weeks_on_chart: 10,
                }],
            },
            Chart {
                data: vec![ChartSong {
                    song: "blinding lights".into(),
                    artist: "the weeknd".into(),
                    peak_position: 1,
                    weeks_on_chart: 90,
                }],
            },
        ];
        assert_eq!(
            unique_songs(&charts),
            vec![Song {
                artist: "The Weeknd".into(),
                title: "Blinding Lights".into(),
                peak: 1,
                weeks: 90,
            }]
        );
    }

    #[test]
    fn query_strips_billboard_credits() {
        assert_eq!(query_artist("Mark Ronson Featuring Bruno Mars"), "Mark Ronson");
        assert_eq!(query_artist("Patti Austin A Duet With James Ingram"), "Patti Austin");
        assert_eq!(query_title("Uptown Funk!"), "Uptown Funk");
        assert_eq!(
            query_title("Gangsta's Paradise (From \"Dangerous Minds\")"),
            "Gangsta's Paradise"
        );
        assert_eq!(
            query_title("Candle In The Wind 1997/Something About The Way You Look Tonight"),
            "Candle In The Wind 1997"
        );
        assert_eq!(query_artist("Puff Daddy (Featuring Mase)"), "Puff Daddy");
        assert_eq!(
            query_artist("The Partridge Family Starring Shirley Jones Featuring David Cassidy"),
            "The Partridge Family"
        );
        assert_eq!(query_artist("Paul McCartney And Wings"), "Paul McCartney");
        assert_eq!(query_artist("Dionne Warwicke & Spinners"), "Dionne Warwicke");
        assert_eq!(query_artist("Elvis Presley With The Jordanaires"), "Elvis Presley");
        assert_eq!(query_artist("Jimmy Gilmer And The Fireballs"), "Jimmy Gilmer");
        assert_eq!(query_artist("Juice WRLD x Marshmello"), "Juice WRLD");
        assert_eq!(query_artist("Nicki Minaj X Lil Baby"), "Nicki Minaj");
        assert_eq!(query_artist("J. Cole, 21 Savage & Morray"), "J. Cole");
        assert_eq!(query_artist("Ryan Hurd With Maren Morris"), "Ryan Hurd");
        assert_eq!(query_title("Coming Up (Live At Glasgow)"), "Coming Up");
        assert_eq!(query_title("Chariots Of Fire - Titles"), "Chariots Of Fire");
    }

    #[test]
    fn plain_from_lrc_drops_timestamps_and_credits() {
        let lrc = "[00:00.00] 作词 : Future\n[00:01.00] I still thank God you woke up\n[00:05.00] 纯音乐\n";
        assert_eq!(plain_from_lrc(lrc).as_deref(), Some("I still thank God you woke up"));
        assert_eq!(plain_from_lrc("[00:00.00] 作曲 : Vangelis\n[00:01.00] 纯音乐\n"), None);
        assert_eq!(
            plain_from_lrc("[unclosed timestamp\n[00:01.00] still here\n").as_deref(),
            Some("[unclosed timestamp\nstill here")
        );
    }

    #[test]
    fn pick_lyrics_skips_instrumental_hits() {
        let hits = [
            LrcHit { plain_lyrics: None },
            LrcHit {
                plain_lyrics: Some("  You can stand under my umbrella  ".into()),
            },
        ];
        assert_eq!(pick_lyrics(&hits).as_deref(), Some("You can stand under my umbrella"));
    }
}
