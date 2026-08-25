use anyhow::{Context, Result};
use reqwest::Url;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 120;

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    format: Value,
    messages: [OllamaMessage<'a>; 1],
    model: &'a str,
    stream: bool,
    think: bool,
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    content: &'a str,
    role: &'static str,
}

#[derive(serde::Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
}

#[derive(serde::Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

#[derive(Debug)]
pub struct OllamaClient {
    chat_url: Url,
    http: reqwest::Client,
    model: String,
}

impl OllamaClient {
    pub fn new(host: &str, model: &str) -> Result<Self> {
        let model = model.trim();
        if model.is_empty() {
            anyhow::bail!("OLLAMA_MODEL is empty");
        }
        let base = Url::parse(host).map_err(|error| anyhow::anyhow!("invalid OLLAMA_HOST '{host}': {error}"))?;
        let chat_url = base
            .join("/api/chat")
            .map_err(|error| anyhow::anyhow!("invalid OLLAMA_HOST '{host}': {error}"))?;
        Self::with_chat_url(chat_url, model)
    }

    pub fn with_chat_url(chat_url: Url, model: &str) -> Result<Self> {
        Ok(Self {
            chat_url,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
                .build()?,
            model: model.to_string(),
        })
    }

    pub async fn complete<T: DeserializeOwned>(&self, prompt: &str, schema: &str) -> Result<T> {
        let format: Value = serde_json::from_str(schema).context("invalid JSON schema")?;
        let request = OllamaChatRequest {
            format,
            messages: [OllamaMessage {
                content: prompt,
                role: "user",
            }],
            model: &self.model,
            stream: false,
            think: false,
        };
        let response = self
            .http
            .post(self.chat_url.clone())
            .json(&request)
            .send()
            .await
            .with_context(|| "ollama request failed")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("ollama returned {status}: {body}");
        }
        let payload: OllamaChatResponse = response.json().await.context("ollama response was not JSON")?;
        parse_model_json(&payload.message.content)
    }
}

pub fn parse_model_json<T: DeserializeOwned>(content: &str) -> Result<T> {
    let json = extract_json_payload(content)
        .ok_or_else(|| anyhow::anyhow!("ollama response did not contain JSON: {content}"))?;
    serde_json::from_str(json).with_context(|| format!("failed to parse ollama JSON: {json}"))
}

fn extract_json_payload(content: &str) -> Option<&str> {
    let without_think = content.rsplit_once("</think>").map_or(content, |(_, rest)| rest);
    let trimmed = without_think.trim();
    let unfenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|value| value.strip_suffix("```").unwrap_or(value).trim())
        .unwrap_or(trimmed);
    if serde_json::from_str::<Value>(unfenced).is_ok() {
        return Some(unfenced);
    }
    let start = unfenced.find(['{', '['])?;
    let end = unfenced.rfind(['}', ']'])?;
    Some(&unfenced[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use serde::Deserialize;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct MatchVerdict {
        matches: bool,
    }

    #[test]
    fn test_parse_model_json_reads_matches() {
        assert_eq!(
            parse_model_json::<MatchVerdict>(r#"{"matches":false}"#).unwrap(),
            MatchVerdict { matches: false }
        );
    }

    #[test]
    fn test_ollama_client_rejects_empty_model() {
        let error = OllamaClient::new("http://127.0.0.1:11434", "  ").unwrap_err();
        assert!(error.to_string().contains("OLLAMA_MODEL"));
    }

    #[tokio::test]
    async fn test_complete_parses_schema_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(r#"{"message":{"content":"{\"matches\":true}"}}"#, "application/json"),
            )
            .mount(&server)
            .await;
        let chat_url = Url::parse(&format!("{}/api/chat", server.uri())).unwrap();
        let client = OllamaClient::with_chat_url(chat_url, "qwen3:14b").unwrap();
        assert_eq!(
            client
                .complete::<MatchVerdict>(
                    "prompt",
                    r#"{"type":"object","properties":{"matches":{"type":"boolean"}}}"#
                )
                .await
                .unwrap(),
            MatchVerdict { matches: true }
        );
    }
}
