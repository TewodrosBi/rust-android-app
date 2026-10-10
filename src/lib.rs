use android_activity::{AndroidApp, PollEvent};
use log::{info, LevelFilter};

#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(LevelFilter::Info)
            .with_tag("RustApp"),
    );

    info!("Rust Native App has booted successfully!");

    loop {
        app.poll_events(Some(std::time::Duration::from_millis(16)), |_event| {});
    }
}
