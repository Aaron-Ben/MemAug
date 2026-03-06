use anyhow::{anyhow, Context, Result};
use glob::glob;
use lazy_static::lazy_static;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy::{doc, Index, tokenizer::BoxTokenStream};
use tokio::fs;
use futures::future::join_all;

// --- Jieba Tokenizer for Tantivy ---

#[derive(Clone)]
pub struct JiebaTokenizer {
    jieba: Arc<jieba_rs::Jieba>,
}

pub struct JiebaTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl Tokenizer for JiebaTokenizer {
    fn token_stream<'a>(&self, text: &'a str) -> BoxTokenStream<'a> {
        let mut tokens = Vec::new();
        let mut offset = 0;

        for word in self.jieba.cut(text, false) {
            // Find the word starting from current offset to handle repeated words correctly
            if let Some(pos) = text[offset..].find(word) {
                let start = offset + pos;
                let end = start + word.len();

                tokens.push(Token {
                    offset_from: start,
                    offset_to: end,
                    position: tokens.len(),
                    text: word.to_string(),
                    position_length: 1,
                });

                offset = end;
            } else {
                // If word not found (shouldn't happen with jieba), skip it
                // This prevents infinite loop if jieba returns unexpected results
                let word_bytes = word.len();
                if word_bytes > 0 {
                    offset += word_bytes;
                }
            }
        }

        let stream: Box<dyn TokenStream + 'a> = Box::new(JiebaTokenStream { tokens, index: 0 });
        stream.into()
    }
}

impl TokenStream for JiebaTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

// --- Lazy Static Regex Cache for Performance ---
lazy_static! {
    static ref STYLE_RE: Regex = Regex::new(r"(?is)<style[^>]*>.*?</style>")
        .expect("Failed to compile STYLE_RE regex");
    static ref SCRIPT_RE: Regex = Regex::new(r"(?is)<script[^>]*>.*?</script>")
        .expect("Failed to compile SCRIPT_RE regex");
    static ref NAKED_CSS_RE: Regex = Regex::new(
        r"(?s)@keyframes[\s\S]*?(\r?\n\r?\n|<[a-zA-Z]|$)"
    ).expect("Failed to compile NAKED_CSS_RE regex");
    static ref CSS_RULE_RE: Regex = Regex::new(
        r"(?m)^\s*[\w\-\.#@][\w\-\s,\.#>+~:()]*\{[^}]*\}"
    ).expect("Failed to compile CSS_RULE_RE regex");
    static ref WHITESPACE_RE: Regex = Regex::new(r"\s+")
        .expect("Failed to compile WHITESPACE_RE regex");
}

// --- Structs for Serialization/Deserialization & Data Handling ---

// Custom deserializer to handle window_size being a string or an integer
fn deserialize_window_size_from_string_or_int<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrInt {
        String(String),
        Int(i32),
    }

    match StringOrInt::deserialize(deserializer)? {
        StringOrInt::String(s) => s.parse::<i32>().map_err(serde::de::Error::custom),
        StringOrInt::Int(i) => Ok(i),
    }
}

#[derive(Deserialize, Debug)]
struct ToolArgs {
    maid: String,
    #[serde(alias = "key_word", alias = "KeyWord")]
    keyword: String,
    #[serde(
        alias = "windowsize",
        default = "default_window_size",
        deserialize_with = "deserialize_window_size_from_string_or_int"
    )]
    window_size: i32,
}

fn default_window_size() -> i32 {
    3
}

#[derive(Serialize, Debug)]
struct SuccessResponse {
    status: &'static str,
    result: String,
}

#[derive(Serialize, Debug)]
struct ErrorResponse {
    status: &'static str,
    error: String,
}

#[derive(Debug, Clone)]
struct Config {
    vchat_data_url: PathBuf,
    max_memo_tokens: usize,
    rerank_search: bool,
    rerank_url: String,
    rerank_api: String,
    rerank_model: String,
    rerank_max_tokens_per_batch: usize,
    rerank_top_n: usize,
    query_preset: String,
}

#[derive(Deserialize, Debug)]
struct AgentConfig {
    name: String,
}

#[derive(Deserialize, Debug)]
struct UserSettings {
    #[serde(default = "default_user_name", rename = "userName")]
    user_name: String,
}
fn default_user_name() -> String {
    "主人".to_string()
}


#[derive(Deserialize, Debug, Clone)]
struct HistoryEntry {
    role: String,
    content: String,
}

// --- AI-Powered Query Parser ---

fn parse_ai_query_to_tantivy(query: &str) -> String {
    let parts = query.split(|c| c == ',' || c == '，')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty());

    let mut tantivy_parts = Vec::new();

    for part in parts {
        // Check for exact phrase: "exact phrase"
        if part.starts_with('"') && part.ends_with('"') {
            tantivy_parts.push(part.to_string());
            continue;
        }
        // Check for weighted term: (term:weight)
        if part.starts_with('(') && part.ends_with(')') {
            if let Some(inner) = part.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
                let mut split = inner.rsplitn(2, ':');
                if let (Some(weight_str), Some(term)) = (split.next(), split.next()) {
                     tantivy_parts.push(format!("{}^{}", term.trim(), weight_str.trim()));
                     continue;
                }
            }
        }
        // Check for negated term: [term:weight]
        if part.starts_with('[') && part.ends_with(']') {
             if let Some(inner) = part.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                let term = inner.split(':').next().unwrap_or("").trim();
                if !term.is_empty() {
                    tantivy_parts.push(format!("-{}", term));
                }
                continue;
            }
        }
        // Check for OR group: {term1|term2} or {term1|term2:weight}
        if part.starts_with('{') && part.ends_with('}') {
            if let Some(inner) = part.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let mut weight_str: Option<&str> = None;
                let mut terms_part = inner;

                if let Some(pos) = inner.rfind(':') {
                    // To avoid splitting a term that contains a colon, check if the part after the colon is a valid number.
                    let (potential_terms, potential_weight) = inner.split_at(pos);
                    let potential_weight_val = &potential_weight[1..];
                    if potential_weight_val.trim().parse::<f32>().is_ok() {
                        terms_part = potential_terms;
                        weight_str = Some(potential_weight_val);
                    }
                }

                let terms: Vec<&str> = terms_part.split('|').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();

                if !terms.is_empty() {
                    let group_query = format!("({})", terms.join(" OR "));
                    if let Some(w_str) = weight_str {
                        tantivy_parts.push(format!("{}^{}", group_query, w_str.trim()));
                    } else {
                        tantivy_parts.push(group_query);
                    }
                }
                continue;
            }
        }
        
        // Normal term
        tantivy_parts.push(part.to_string());
    }

    tantivy_parts.join(" ")
}

#[derive(Debug)]
struct AgentInfo {
    name: String,
    uuid: String,
}

#[derive(Serialize, Debug)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    return_documents: bool,
    top_n: usize,
}

#[derive(Deserialize, Debug)]
struct RerankResult {
    index: usize,
    #[allow(dead_code)]
    relevance_score: f64,
}

#[derive(Deserialize, Debug)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

// --- Main Application Logic ---

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        let error_response = ErrorResponse {
            status: "error",
            error: format!("[DeepMemo-rs] {:?}", e),
        };
        if let Ok(json_err) = serde_json::to_string(&error_response) {
            eprintln!("{}", json_err);
        } else {
            eprintln!("{{\"status\":\"error\",\"error\":\"[DeepMemo-rs] Failed to serialize error message.\"}}");
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let config = load_config().context("Failed to load configuration")?;
    let input_json = read_stdin().context("Failed to read from stdin")?;
    let args: ToolArgs = serde_json::from_str(&input_json)
        .with_context(|| format!("Invalid input format, failed to parse JSON. Input: {}", input_json))?;

    // Combine user keywords with global blocked keywords (now a query preset)
    let final_query = if config.query_preset.is_empty() {
        args.keyword.clone()
    } else if args.keyword.is_empty() {
        config.query_preset.clone()
    } else {
        format!("{},{}", args.keyword, config.query_preset)
    };
    let final_query = final_query.trim();

    if final_query.is_empty() {
        return Err(anyhow!("Query is empty after combining with presets."));
    }

    let agent_info = find_agent_info(&config.vchat_data_url, &args.maid)
        .await?
        .ok_or_else(|| anyhow!("Agent '{}' not found.", args.maid))?;
    let user_name = find_user_name(&config.vchat_data_url).await?;

    let mut memories = search_histories(
        &config.vchat_data_url,
        &agent_info.uuid,
        final_query,
        args.window_size,
        &user_name,
        &agent_info.name,
    )
    .await?;

    if config.rerank_search && !memories.is_empty() {
        eprintln!("[DEBUG] Starting rerank for {} memories...", memories.len());
        memories = match rerank_memories(memories.clone(), final_query, &config).await {
            Ok(m) => {
                eprintln!("[DEBUG] Rerank completed. Got {} memories back.", m.len());
                m
            }
            Err(e) => {
                eprintln!("[DEBUG] Rerank failed: {}. Continuing with original results.", e);
                memories // Fallback to original memories
            }
        };
    }

    let mut output = memories.join("\n\n");
    if output.len() > config.max_memo_tokens {
        let mut new_len = config.max_memo_tokens;
        while !output.is_char_boundary(new_len) {
            new_len -= 1;
        }
        output.truncate(new_len);
        output.push_str("\n... [内容过长，已被截断]");
    }

    if output.trim().is_empty() {
        output = format!("[DeepMemo] 未找到与关键词“{}”相关的回忆。", args.keyword);
    }

    let success_response = SuccessResponse {
        status: "success",
        result: output,
    };
    let output_json = serde_json::to_string(&success_response)?;
    println!("{}", output_json);

    Ok(())
}

// --- Core Logic Functions ---

async fn process_single_history_file(
    file_path: PathBuf,
    query: Arc<String>,
    window_size: i32,
) -> Result<Vec<Vec<HistoryEntry>>> {
    let content = fs::read_to_string(&file_path).await?;
    let history: Vec<HistoryEntry> = match serde_json::from_str::<Vec<HistoryEntry>>(&content) {
        Ok(data) if !data.is_empty() => data,
        _ => return Ok(Vec::new()),
    };

    let schema = {
        let mut schema_builder = Schema::builder();
        let text_indexing_options = TextFieldIndexing::default()
            .set_tokenizer("jieba")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default()
            .set_indexing_options(text_indexing_options)
            .set_stored();
        schema_builder.add_text_field("content", text_options);
        schema_builder.add_u64_field("id", INDEXED | STORED);
        schema_builder.build()
    };

    let index = Index::create_in_ram(schema.clone());
    let jieba_tokenizer = JiebaTokenizer {
        jieba: Arc::new(jieba_rs::Jieba::new()),
    };
    index.tokenizers().register("jieba", jieba_tokenizer);

    let mut index_writer = index.writer(50_000_000)?;
    let content_field = schema
        .get_field("content")
        .expect("Schema should have a 'content' field");
    let id_field = schema
        .get_field("id")
        .expect("Schema should have an 'id' field");

    for (i, entry) in history.iter().enumerate() {
        let clean_content = extract_text(&entry.content);
        if !clean_content.trim().is_empty() {
            index_writer.add_document(doc!(
                content_field => clean_content,
                id_field => i as u64
            ))?;
        }
    }
    index_writer.commit()?;

    let tantivy_query_str = parse_ai_query_to_tantivy(&query);
    eprintln!("[QUERY] Executing Tantivy Query: {}", tantivy_query_str);

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    let mut file_memories = Vec::new();
    let mut used_indices = HashSet::new();

    if let Ok(query) = query_parser.parse_query(&tantivy_query_str) {
        let top_docs = searcher.search(&query, &TopDocs::with_limit(100))?;
        for (_score, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address)?;
            let id_value = retrieved_doc
                .get_first(id_field)
                .expect("Document should have an id field")
                .as_u64()
                .expect("Id field should be a u64");
            let match_index = id_value as usize;

            if used_indices.contains(&match_index) {
                continue;
            }
            
            let start = (match_index as i32 - window_size).max(0) as usize;
            let end = (match_index + window_size as usize + 1).min(history.len());
            let context_slice = &history[start..end];
            file_memories.push(context_slice.to_vec());
            
            for i in start..end {
                used_indices.insert(i);
            }
        }
    }
    
    Ok(file_memories)
}

async fn search_histories(
    chat_path: &Path,
    agent_uuid: &str,
    keywords: &str,
    window_size: i32,
    user_name: &str,
    agent_name: &str,
) -> Result<Vec<String>> {
    let topics_dir = chat_path.join(agent_uuid).join("topics");
    if !topics_dir.exists() {
        return Ok(Vec::new());
    }

    let pattern = topics_dir.join("*.json");
    let mut history_files: Vec<(PathBuf, SystemTime)> = Vec::new();
    for entry in glob(&pattern.to_string_lossy())? {
        if let Ok(path) = entry {
            if let Ok(metadata) = fs::metadata(&path).await {
                history_files.push((path, metadata.modified()?));
            }
        }
    }

    history_files.sort_by_key(|k| k.1);
    history_files.reverse();

    let query_arc = Arc::new(keywords.to_string());
    let mut tasks = Vec::new();

    for (file_path, _) in history_files.into_iter().skip(1) {
        let query_clone = Arc::clone(&query_arc);
        tasks.push(tokio::spawn(async move {
            process_single_history_file(file_path, query_clone, window_size).await
        }));
    }

    let results = join_all(tasks).await;

    let mut all_memories = Vec::new();
    let mut memory_index = 1;

    for result in results {
        match result {
            Ok(Ok(file_memories)) => {
                for context_slice in file_memories {
                    if let Some(formatted_memory) =
                        format_memory(&context_slice, user_name, agent_name, memory_index)
                    {
                        all_memories.push(formatted_memory);
                        memory_index += 1;
                    }
                }
            }
            Ok(Err(e)) => eprintln!("[ERROR] Failed to process a history file: {}", e),
            Err(e) => eprintln!("[ERROR] A task panicked while processing history file: {}", e),
        }
    }

    Ok(all_memories)
}

// --- Rerank Logic ---

#[derive(Debug)]
struct ScoredDocument {
    document: String,
    score: f64,
}

async fn rerank_memories(memories: Vec<String>, query: &str, config: &Config) -> Result<Vec<String>> {
    if config.rerank_url.is_empty() || memories.is_empty() {
        eprintln!("[DEBUG] Rerank skipped: URL not configured or no memories to rerank.");
        return Ok(memories);
    }

    eprintln!("[DEBUG] Starting batch rerank for {} memories...", memories.len());
    let batches = create_batches(&memories, config.rerank_max_tokens_per_batch);
    eprintln!("[DEBUG] Created {} batches.", batches.len());

    // Track if any batch succeeded
    let mut any_batch_succeeded = false;
    let mut all_batches_failed = true;

    let mut tasks = Vec::new();
    for batch in batches {
        let q = query.to_string();
        let cfg = config.clone();
        tasks.push(tokio::spawn(async move {
            perform_rerank_request(&batch, &q, &cfg).await
        }));
    }

    let mut all_scored_results: Vec<ScoredDocument> = Vec::new();
    for task in tasks {
        match task.await {
            Ok(Ok(scored_batch_results)) => {
                if !scored_batch_results.is_empty() {
                    any_batch_succeeded = true;
                    all_batches_failed = false;
                    all_scored_results.extend(scored_batch_results);
                }
            }
            Ok(Err(e)) => {
                eprintln!("[DEBUG] A rerank batch failed: {}", e);
            }
            Err(e) => {
                eprintln!("[DEBUG] A rerank task panicked: {}", e);
            }
        }
    }
    eprintln!("[DEBUG] Collected {} scored results from all batches.", all_scored_results.len());

    // Fallback to original results if all batches failed or no results were returned
    if all_batches_failed || (!any_batch_succeeded && !memories.is_empty()) {
        eprintln!("[WARNING] All rerank batches failed or returned no results, falling back to original top N.");
        return Ok(memories.into_iter().take(config.rerank_top_n).collect());
    }

    all_scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let final_memories: Vec<String> = all_scored_results
        .into_iter()
        .take(config.rerank_top_n)
        .map(|res| res.document)
        .collect();

    eprintln!("[DEBUG] Rerank finished. Returning top {} memories.", final_memories.len());

    // Additional fallback: if rerank returned empty results despite some batches succeeding
    if final_memories.is_empty() && !memories.is_empty() {
        eprintln!("[WARNING] Rerank resulted in zero memories, falling back to original top N.");
        return Ok(memories.into_iter().take(config.rerank_top_n).collect());
    }

    Ok(final_memories)
}

fn create_batches(documents: &[String], max_tokens: usize) -> Vec<Vec<String>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_tokens = 0;
    
    for doc in documents {
        let doc_len = doc.len();
        
        let processed_doc = if doc_len > max_tokens * 8 / 10 {
            let keep_length = max_tokens / 4;
            if keep_length * 2 >= doc.len() {
                // 如果文档不够长，无法安全地截取头尾，则不处理
                doc.clone()
            } else {
                let mut head_end = keep_length;
                while head_end > 0 && !doc.is_char_boundary(head_end) {
                    head_end -= 1;
                }

                let mut tail_start = doc.len().saturating_sub(keep_length);
                while tail_start < doc.len() && !doc.is_char_boundary(tail_start) {
                    tail_start += 1;
                }
                
                if head_end >= tail_start {
                    // 如果头尾重叠，返回原文案全截断
                    let mut safe_len = max_tokens * 8 / 10;
                    while safe_len > 0 && !doc.is_char_boundary(safe_len) {
                        safe_len -= 1;
                    }
                    doc[..safe_len].to_string()
                } else {
                    format!(
                        "{}...[内容过长，中间已截断]...{}",
                        &doc[..head_end],
                        &doc[tail_start..]
                    )
                }
            }
        } else {
            doc.clone()
        };
        
        let processed_len = processed_doc.len();
        
        if !current_batch.is_empty()
            && (current_tokens + processed_len > max_tokens || current_batch.len() >= 15) {
            batches.push(current_batch);
            current_batch = Vec::new();
            current_tokens = 0;
        }
        
        current_batch.push(processed_doc);
        current_tokens += processed_len;
    }
    
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }
    batches
}

async fn perform_rerank_request(documents: &[String], query: &str, config: &Config) -> Result<Vec<ScoredDocument>> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }
    let client = reqwest::Client::new();
    let rerank_endpoint = format!("{}v1/rerank", config.rerank_url);
    let request_body = RerankRequest {
        model: &config.rerank_model,
        query,
        documents,
        return_documents: false,
        top_n: documents.len(),
    };
    let response = client
        .post(&rerank_endpoint)
        .bearer_auth(&config.rerank_api)
        .json(&request_body)
        .send()
        .await?
        .json::<RerankResponse>()
        .await?;

    Ok(response
        .results
        .into_iter()
        .filter_map(|res| {
            documents.get(res.index).map(|doc| ScoredDocument {
                document: doc.clone(),
                score: res.relevance_score,
            })
        })
        .collect())
}

// --- Helper & Utility Functions ---

async fn find_agent_info(chat_path: &Path, maid_name: &str) -> Result<Option<AgentInfo>> {
    // 直接在 chat_path 下查找角色目录
    let mut read_dir = fs::read_dir(chat_path).await?;
    while let Some(entry) = read_dir.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            // 尝试从 prompt.md 读取角色名
            let prompt_path = path.join("prompt.md");
            if let Ok(content) = fs::read_to_string(&prompt_path).await {
                // 从 prompt.md 第一行提取角色名 (格式: # 角色名)
                if let Some(first_line) = content.lines().next() {
                    let name = first_line.trim_start_matches('#').trim();
                    let dir_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");
                    if !name.is_empty() && (name.contains(maid_name) || maid_name.contains(name) || dir_name == maid_name) {
                        return Ok(Some(AgentInfo {
                            name: name.to_string(),
                            uuid: entry.file_name().to_string_lossy().into_owned(),
                        }));
                    }
                }
            }
        }
    }
    Ok(None)
}

async fn find_user_name(_chat_path: &Path) -> Result<String> {
    // 默认返回 "主人"，如果需要可以从其他配置读取
    Ok("主人".to_string())
}

fn format_memory(slice: &[HistoryEntry], user_name: &str, agent_name: &str, memory_index: usize) -> Option<String> {
    let memory_string: String = slice
        .iter()
        .filter_map(|entry| {
            let name = if entry.role == "user" {
                user_name
            } else if entry.role == "assistant" {
                agent_name
            } else {
                return None;
            };
            let clean_content = extract_text(&entry.content);
            if clean_content.is_empty() {
                None
            } else {
                Some(format!("{}: {}", name, clean_content))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if memory_string.is_empty() {
        None
    } else {
        Some(format!("[回忆片段{}]:\n{}", memory_index, memory_string))
    }
}

fn extract_text(html: &str) -> String {
    // Using the advanced multi-step cleaning logic provided by the user.
    let mut clean = html.to_string();

    // 1. 移除 <style> 标签
    clean = STYLE_RE.replace_all(&clean, "").to_string();

    // 2. 移除 <script> 标签
    clean = SCRIPT_RE.replace_all(&clean, "").to_string();

    // 3. 【核心】移除裸露 CSS 块
    // 匹配从第一个 CSS 规则开始，到双换行或 HTML 标签为止
    // 这会匹配：@keyframes ... } .class { ... } .another { ... } 直到遇到空行
    clean = NAKED_CSS_RE.replace_all(&clean, "$1").to_string();

    // 4. 兜底：移除任何剩余的 CSS 规则块
    clean = CSS_RULE_RE.replace_all(&clean, "").to_string();

    // 5. 清理 HTML
    let result = ammonia::clean(&clean);

    // 6. 清理空白
    WHITESPACE_RE.replace_all(result.trim(), " ").to_string()
}


fn read_stdin() -> Result<String> {
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

fn load_config() -> Result<Config> {
    let exe_dir = std::env::current_exe()?
        .parent()
        .expect("Executable path should have a parent directory")
        .to_path_buf();
    let candidates = vec![
        exe_dir.join("config.env"),
        exe_dir.join("..").join("config.env"),
        exe_dir.join("..").join("..").join("config.env"),
    ];
    let config_path = candidates
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("config.env not found in search paths"))?;
    
    dotenv::from_path(&config_path).with_context(|| format!("Failed to load .env file from {:?}", config_path))?;

    let vchat_data_url = match std::env::var("VchatDataURL") {
        Ok(url) if !url.trim().is_empty() => PathBuf::from(url),
        _ => {
            // config_path is VCPDistributedServer/Plugin/DeepMemo/config.env
            // Project root (h:/MCP/VCPChat) is 4 levels up from config_path.
            let project_root = config_path
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .and_then(|p| p.parent())
                .ok_or_else(|| anyhow!("Failed to determine project root from config path."))?;

            project_root.join("AppData")
        }
    };

    Ok(Config {
        vchat_data_url,
        max_memo_tokens: std::env::var("MaxMemoTokens")?.parse()?,
        rerank_search: std::env::var("RerankSearch")?.to_lowercase() == "true",
        rerank_url: std::env::var("RerankUrl")?,
        rerank_api: std::env::var("RerankApi")?,
        rerank_model: std::env::var("RerankModel")?,
        rerank_max_tokens_per_batch: std::env::var("RerankMaxTokensPerBatch")?.parse()?,
        rerank_top_n: std::env::var("RerankTopN")?.parse()?,
        query_preset: std::env::var("QueryPreset").unwrap_or_default(),
    })
}