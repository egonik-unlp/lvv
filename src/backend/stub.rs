//! A tiny HTTP/1.1 server for testing the real clients without a model server.

use std::sync::{Arc, Mutex};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

/// A canned HTTP response.
#[derive(Clone)]
pub(crate) struct StubResponse {
    status: u16,
    retry_after: Option<String>,
    body: String,
}

impl StubResponse {
    pub(crate) fn status(status: u16) -> Self {
        StubResponse {
            status,
            retry_after: None,
            body: String::new(),
        }
    }

    pub(crate) fn ok(body: impl Into<String>) -> Self {
        StubResponse::status(200).body(body)
    }

    pub(crate) fn body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    pub(crate) fn retry_after(mut self, seconds: Option<&str>) -> Self {
        self.retry_after = seconds.map(str::to_string);
        self
    }
}

/// A request the stub received.
#[derive(Debug, Clone)]
pub(crate) struct StubRequest {
    pub(crate) path: String,
    pub(crate) authorization: Option<String>,
    pub(crate) body: serde_json::Value,
}

/// Answers each connection with the next scripted response. The last response
/// repeats once the script runs out.
pub(crate) struct StubServer {
    addr: std::net::SocketAddr,
    requests: Arc<Mutex<Vec<StubRequest>>>,
}

impl StubServer {
    pub(crate) async fn start(responses: Vec<StubResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        tokio::spawn(async move {
            let mut served = 0;
            while let Ok((stream, _)) = listener.accept().await {
                let response = responses[served.min(responses.len() - 1)].clone();
                served += 1;
                if let Some(request) = serve(stream, response).await {
                    recorded.lock().unwrap().push(request);
                }
            }
        });
        StubServer { addr, requests }
    }

    /// Accepts connections and never answers.
    pub(crate) async fn silent() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut open = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                open.push(stream);
            }
        });
        StubServer {
            addr,
            requests: Arc::default(),
        }
    }

    pub(crate) fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    pub(crate) fn requests(&self) -> Vec<StubRequest> {
        self.requests.lock().unwrap().clone()
    }
}

/// A URL on a port nothing listens on.
pub(crate) async fn closed_port_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}/x")
}

async fn serve(mut stream: TcpStream, response: StubResponse) -> Option<StubRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let header = |name: &str| {
        head.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name)
                .then(|| value.trim().to_string())
        })
    };
    let length: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buffer.len() < header_end + length {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
    let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
    let body = serde_json::from_slice(&buffer[header_end..]).unwrap_or(serde_json::Value::Null);
    let request = StubRequest {
        path,
        authorization: header("authorization"),
        body,
    };

    let mut reply = format!(
        "HTTP/1.1 {} Stub\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        response.status,
        response.body.len()
    );
    if let Some(seconds) = &response.retry_after {
        reply.push_str(&format!("Retry-After: {seconds}\r\n"));
    }
    reply.push_str("\r\n");
    reply.push_str(&response.body);
    stream.write_all(reply.as_bytes()).await.ok()?;
    stream.shutdown().await.ok();
    Some(request)
}
