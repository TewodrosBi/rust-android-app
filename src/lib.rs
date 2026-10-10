use std::sync::mpsc;
use eframe::egui;
use egui::{Color32, RichText, Rounding, Stroke};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ApiProvider {
    Groq,
    Gemini,
}

impl ApiProvider {
    pub fn name(&self) -> &'static str {
        match self {
            ApiProvider::Groq => "Groq (Llama 3.3)",
            ApiProvider::Gemini => "Google (Gemini 1.5 Flash)",
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User,
    Assistant,
    Error,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum WorkerEvent {
    Response(String),
    Error(String),
}

pub struct ChatApp {
    provider: ApiProvider,
    groq_api_key: String,
    gemini_api_key: String,
    show_api_key: bool,
    messages: Vec<ChatMessage>,
    input_text: String,
    is_generating: bool,
    error_banner: Option<String>,
    tx: mpsc::Sender<WorkerEvent>,
    rx: mpsc::Receiver<WorkerEvent>,
}

impl ChatApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            provider: ApiProvider::Groq,
            groq_api_key: String::new(),
            gemini_api_key: String::new(),
            show_api_key: false,
            messages: vec![ChatMessage {
                role: Role::Assistant,
                content: "Hello! I am your mobile Rust AI Chat assistant. Choose your provider above, enter your API key, and ask any question!".to_string(),
            }],
            input_text: String::new(),
            is_generating: false,
            error_banner: None,
            tx,
            rx,
        }
    }

    fn send_current_message(&mut self, ctx: &egui::Context) {
        let user_text = self.input_text.trim().to_string();
        if user_text.is_empty() || self.is_generating {
            return;
        }

        let api_key = match self.provider {
            ApiProvider::Groq => self.groq_api_key.trim().to_string(),
            ApiProvider::Gemini => self.gemini_api_key.trim().to_string(),
        };

        if api_key.is_empty() {
            self.error_banner = Some(format!(
                "Please enter an API key for {}",
                self.provider.name()
            ));
            return;
        }

        self.error_banner = None;
        self.input_text.clear();
        self.messages.push(ChatMessage {
            role: Role::User,
            content: user_text,
        });

        self.is_generating = true;

        let provider = self.provider;
        let history = self.messages.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            let res = call_ai_api(provider, &api_key, &history);
            match res {
                Ok(reply) => {
                    let _ = tx.send(WorkerEvent::Response(reply));
                }
                Err(err) => {
                    let _ = tx.send(WorkerEvent::Error(err));
                }
            }
            ctx.request_repaint();
        });
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain any incoming responses from the background worker thread
        while let Ok(event) = self.rx.try_recv() {
            self.is_generating = false;
            match event {
                WorkerEvent::Response(reply) => {
                    self.error_banner = None;
                    self.messages.push(ChatMessage {
                        role: Role::Assistant,
                        content: reply,
                    });
                }
                WorkerEvent::Error(err) => {
                    self.error_banner = Some(err.clone());
                    self.messages.push(ChatMessage {
                        role: Role::Error,
                        content: format!("Error: {}", err),
                    });
                }
            }
        }

        // Top Panel: API Provider selection and API Key entry
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("🤖 Rust AI Chat");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("🗑 Clear").on_hover_text("Clear conversation history").clicked() {
                        self.messages.clear();
                        self.error_banner = None;
                    }
                });
            });

            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Provider:").strong());
                ui.radio_value(&mut self.provider, ApiProvider::Groq, "Groq (Llama 3.3)");
                ui.radio_value(
                    &mut self.provider,
                    ApiProvider::Gemini,
                    "Google (Gemini 1.5 Flash)",
                );
            });

            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("API Key:").strong());
                let key_ref = match self.provider {
                    ApiProvider::Groq => &mut self.groq_api_key,
                    ApiProvider::Gemini => &mut self.gemini_api_key,
                };
                let edit = egui::TextEdit::singleline(key_ref)
                    .password(!self.show_api_key)
                    .hint_text(match self.provider {
                        ApiProvider::Groq => "Enter Groq API Key (gsk_...)",
                        ApiProvider::Gemini => "Enter Gemini API Key (AIza...)",
                    })
                    .desired_width(ui.available_width() - 36.0);
                ui.add(edit);

                let toggle_icon = if self.show_api_key { "🔒" } else { "👁" };
                if ui
                    .button(toggle_icon)
                    .on_hover_text("Toggle show/hide API key")
                    .clicked()
                {
                    self.show_api_key = !self.show_api_key;
                }
            });

            if let Some(err) = &self.error_banner {
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(Color32::from_rgb(80, 20, 20))
                    .rounding(Rounding::same(4.0))
                    .inner_margin(egui::Margin::same(6.0))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("⚠ {}", err))
                                .color(Color32::from_rgb(255, 180, 180)),
                        );
                    });
            }

            if self.is_generating {
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new("Generating response...").italics());
                });
            }

            ui.add_space(4.0);
        });

        // Bottom Panel: Input bar and Send button
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let text_input = egui::TextEdit::singleline(&mut self.input_text)
                    .hint_text("Ask anything... (Enter to send)")
                    .desired_width(ui.available_width() - 85.0);

                let response = ui.add(text_input);

                let can_send = !self.is_generating && !self.input_text.trim().is_empty();
                let send_btn = ui.add_enabled(
                    can_send,
                    egui::Button::new(RichText::new("Send 🚀").strong()),
                );

                let enter_pressed = response.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);

                if (send_btn.clicked() || enter_pressed) && can_send {
                    self.send_current_message(ctx);
                    response.request_focus();
                }
            });
            ui.add_space(6.0);
        });

        // Central Panel: Scrollable conversation history
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    for msg in &self.messages {
                        render_chat_bubble(ui, msg);
                        ui.add_space(6.0);
                    }
                    if self.is_generating {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                RichText::new("Thinking...")
                                    .italics()
                                    .color(Color32::LIGHT_GRAY),
                            );
                        });
                    }
                });
        });
    }
}

fn render_chat_bubble(ui: &mut egui::Ui, msg: &ChatMessage) {
    let (bg_color, stroke_color, role_label, role_color, text_color) = match msg.role {
        Role::User => (
            Color32::from_rgb(30, 45, 70),
            Color32::from_rgb(60, 90, 140),
            "👤 You",
            Color32::from_rgb(130, 180, 255),
            Color32::WHITE,
        ),
        Role::Assistant => (
            Color32::from_rgb(35, 38, 44),
            Color32::from_rgb(65, 70, 80),
            "🤖 Assistant",
            Color32::from_rgb(140, 220, 160),
            Color32::from_rgb(230, 230, 230),
        ),
        Role::Error => (
            Color32::from_rgb(60, 25, 25),
            Color32::from_rgb(120, 50, 50),
            "⚠️ Error",
            Color32::from_rgb(255, 120, 120),
            Color32::from_rgb(255, 200, 200),
        ),
    };

    egui::Frame::none()
        .fill(bg_color)
        .stroke(Stroke::new(1.0, stroke_color))
        .rounding(Rounding::same(8.0))
        .inner_margin(egui::Margin::same(10.0))
        .show(ui, |ui| {
            ui.label(RichText::new(role_label).strong().color(role_color).size(12.0));
            ui.add_space(2.0);
            ui.label(RichText::new(&msg.content).color(text_color).size(14.0));
        });
}

fn call_ai_api(
    provider: ApiProvider,
    api_key: &str,
    history: &[ChatMessage],
) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

    match provider {
        ApiProvider::Groq => {
            let messages: Vec<serde_json::Value> = history
                .iter()
                .filter_map(|m| match m.role {
                    Role::User => Some(serde_json::json!({
                        "role": "user",
                        "content": m.content
                    })),
                    Role::Assistant => Some(serde_json::json!({
                        "role": "assistant",
                        "content": m.content
                    })),
                    Role::Error => None,
                })
                .collect();

            let payload = serde_json::json!({
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048
            });

            let resp = client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .map_err(|e| format!("Network request failed: {}", e))?;

            let status = resp.status();
            let body_text = resp
                .text()
                .map_err(|e| format!("Failed reading response: {}", e))?;

            if !status.is_success() {
                return Err(format!("Groq API error ({status}): {body_text}"));
            }

            let parsed: serde_json::Value = serde_json::from_str(&body_text)
                .map_err(|e| format!("Failed to parse response JSON: {}", e))?;

            let content = parsed["choices"][0]["message"]["content"]
                .as_str()
                .ok_or_else(|| format!("Unexpected response format: {body_text}"))?;

            Ok(content.to_string())
        }
        ApiProvider::Gemini => {
            let contents: Vec<serde_json::Value> = history
                .iter()
                .filter_map(|m| match m.role {
                    Role::User => Some(serde_json::json!({
                        "role": "user",
                        "parts": [{ "text": m.content }]
                    })),
                    Role::Assistant => Some(serde_json::json!({
                        "role": "model",
                        "parts": [{ "text": m.content }]
                    })),
                    Role::Error => None,
                })
                .collect();

            let payload = serde_json::json!({
                "contents": contents
            });

            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={}",
                api_key
            );

            let resp = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .map_err(|e| format!("Network request failed: {}", e))?;

            let status = resp.status();
            let body_text = resp
                .text()
                .map_err(|e| format!("Failed reading response: {}", e))?;

            if !status.is_success() {
                return Err(format!("Gemini API error ({status}): {body_text}"));
            }

            let parsed: serde_json::Value = serde_json::from_str(&body_text)
                .map_err(|e| format!("Failed to parse response JSON: {}", e))?;

            let content = parsed["candidates"][0]["content"]["parts"][0]["text"]
                .as_str()
                .ok_or_else(|| format!("Unexpected response format: {body_text}"))?;

            Ok(content.to_string())
        }
    }
}

// Android native entry point
#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("RustAIChat"),
    );

    log::info!("Rust AI Chat has booted successfully!");

    let native_options = eframe::NativeOptions {
        android_app: Some(app),
        ..Default::default()
    };

    let _ = eframe::run_native(
        "Rust AI Chat",
        native_options,
        Box::new(|cc| Ok(Box::new(ChatApp::new(cc)))),
    );
}
