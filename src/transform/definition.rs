use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::error::{CompletionError, InputError};
use crate::backend::OutputSchema;

type InputFn<T> = dyn Fn(&T) -> Result<String, InputError> + Send + Sync;
type ApplyFn<T, O> = dyn Fn(&mut T, O) + Send + Sync;

/// How to parse a reply into `O`, and how to store `O` in the cache.
pub(crate) struct OutputSpec<O> {
    pub(crate) schema: Option<OutputSchema>,
    pub(crate) parse: fn(&str) -> Result<O, CompletionError>,
    pub(crate) encode: fn(&O) -> Option<Value>,
    pub(crate) decode: fn(Value) -> Option<O>,
}

/// One task for a chat model over records of type `T`: the instructions,
/// what to send for each record, and where the reply goes.
///
/// Build one with [`text`](Self::text) for a plain-text reply or
/// [`structured`](Self::structured) for a typed JSON reply, then run it with
/// [`Llm::run`](super::Llm::run) or [`Llm::complete`](super::Llm::complete).
/// A transform doesn't hold a model, so the same transform runs against any
/// [`Llm`](super::Llm), and a record type can have as many transforms as you
/// need.
///
/// # Example
///
/// ```
/// use lvv::transform::Transform;
/// use schemars::JsonSchema;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize)]
/// struct Position {
///     title: String,
///     body: String,
///     summary: Option<String>,
///     keywords: Vec<String>,
/// }
///
/// // Plain text, sending only the body.
/// let summarize = Transform::text("Summarize the job in one sentence.")
///     .input(|p: &Position| p.body.clone())
///     .apply(|p: &mut Position, summary| p.summary = Some(summary));
///
/// // Typed output: the schema of `Tags` goes to the model.
/// #[derive(Serialize, Deserialize, JsonSchema)]
/// struct Tags {
///     keywords: Vec<String>,
/// }
/// let tag = Transform::structured("List the job's search keywords.")
///     .apply(|p: &mut Position, tags: Tags| p.keywords = tags.keywords);
/// assert_eq!(tag.schema().unwrap().name, "Tags");
/// ```
pub struct Transform<T, O = String> {
    prompt: String,
    input: Box<InputFn<T>>,
    pub(crate) output: OutputSpec<O>,
    apply: Option<Box<ApplyFn<T, O>>>,
}

impl<T> Transform<T, String>
where
    T: Serialize + 'static,
{
    /// A transform whose reply is plain text, trimmed of surrounding
    /// whitespace. `prompt` is the system prompt.
    ///
    /// Each record is sent as JSON unless you set [`input`](Self::input).
    pub fn text(prompt: impl Into<String>) -> Self {
        Transform {
            prompt: prompt.into(),
            input: Box::new(json_input::<T>),
            output: OutputSpec {
                schema: None,
                parse: parse_text,
                encode: |text| Some(Value::String(text.clone())),
                decode: |value| match value {
                    Value::String(text) => Some(text),
                    _ => None,
                },
            },
            apply: None,
        }
    }
}

impl<T, O> Transform<T, O>
where
    T: 'static,
    O: 'static,
{
    /// A transform whose reply is JSON of type `O`. `prompt` is the system
    /// prompt.
    ///
    /// The JSON schema of `O` is sent with every request, and each reply is
    /// parsed into `O`. A reply wrapped in a Markdown code fence is accepted.
    /// Each record is sent as JSON unless you set [`input`](Self::input).
    pub fn structured(prompt: impl Into<String>) -> Self
    where
        T: Serialize,
        O: DeserializeOwned + Serialize + JsonSchema,
    {
        Transform {
            prompt: prompt.into(),
            input: Box::new(json_input::<T>),
            output: OutputSpec {
                schema: Some(schema_of::<O>()),
                parse: parse_json::<O>,
                encode: |output| serde_json::to_value(output).ok(),
                decode: |value| serde_json::from_value(value).ok(),
            },
            apply: None,
        }
    }

    /// Sets what is sent for each record, in place of its JSON.
    pub fn input(mut self, input: impl Fn(&T) -> String + Send + Sync + 'static) -> Self {
        self.input = Box::new(move |record| Ok(input(record)));
        self
    }

    /// Sets where the output goes. [`Llm::run`](super::Llm::run) needs it;
    /// [`Llm::complete`](super::Llm::complete) ignores it.
    pub fn apply(mut self, apply: impl Fn(&mut T, O) + Send + Sync + 'static) -> Self {
        self.apply = Some(Box::new(apply));
        self
    }
}

impl<T, O> Transform<T, O> {
    /// The system prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// The schema sent with each request, for structured transforms.
    pub fn schema(&self) -> Option<&OutputSchema> {
        self.output.schema.as_ref()
    }

    pub(crate) fn render(&self, record: &T) -> Result<String, InputError> {
        (self.input)(record)
    }

    pub(crate) fn apply_fn(&self) -> Option<&ApplyFn<T, O>> {
        self.apply.as_deref()
    }
}

fn json_input<T: Serialize>(record: &T) -> Result<String, InputError> {
    Ok(serde_json::to_string(record)?)
}

fn schema_of<O: JsonSchema>() -> OutputSchema {
    let mut schema = schemars::schema_for!(O);
    // Not part of the schema proper, and some servers reject it.
    schema.remove("$schema");
    let name: String = O::schema_name()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    OutputSchema {
        name,
        schema: schema.to_value(),
    }
}

fn parse_text(raw: &str) -> Result<String, CompletionError> {
    let text = raw.trim();
    if text.is_empty() {
        return Err(CompletionError::EmptyResponse);
    }
    Ok(text.to_string())
}

fn parse_json<O: DeserializeOwned>(raw: &str) -> Result<O, CompletionError> {
    let text = raw.trim();
    if text.is_empty() {
        return Err(CompletionError::EmptyResponse);
    }
    match serde_json::from_str(text) {
        Ok(output) => Ok(output),
        Err(source) => strip_code_fence(text)
            .and_then(|inner| serde_json::from_str(inner).ok())
            .ok_or_else(|| CompletionError::Parse {
                raw: raw.to_string(),
                source,
            }),
    }
}

/// The body of a Markdown code fence such as `` ```json\n{..}\n``` ``.
fn strip_code_fence(text: &str) -> Option<&str> {
    let rest = text.strip_prefix("```")?;
    let body = &rest[rest.find('\n')? + 1..];
    Some(body.trim_end().strip_suffix("```")?.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Tags {
        keywords: Vec<String>,
    }

    #[test]
    fn text_is_trimmed_and_empty_is_an_error() {
        assert_eq!(parse_text("  hi \n").unwrap(), "hi");
        assert!(matches!(
            parse_text(" \n"),
            Err(CompletionError::EmptyResponse)
        ));
    }

    #[test]
    fn json_parses_plain_and_fenced() {
        let plain: Tags = parse_json(r#"{"keywords":["rust"]}"#).unwrap();
        let fenced: Tags = parse_json("```json\n{\"keywords\":[\"rust\"]}\n```").unwrap();
        assert_eq!(plain, fenced);
    }

    #[test]
    fn unparseable_json_keeps_the_raw_reply() {
        match parse_json::<Tags>("not json") {
            Err(CompletionError::Parse { raw, .. }) => assert_eq!(raw, "not json"),
            other => panic!("expected a parse error, got {other:?}"),
        }
    }
}
