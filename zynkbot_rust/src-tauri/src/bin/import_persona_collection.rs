use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let mut arguments = std::env::args().skip(1);
    let user_id = arguments.next().unwrap_or_else(|| {
        eprintln!("Usage: import_persona_collection <user-id> <package.json> [--dry-run]");
        std::process::exit(2);
    });
    let package_path = arguments.next().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: import_persona_collection <user-id> <package.json> [--dry-run]");
        std::process::exit(2);
    });
    let dry_run = arguments.any(|argument| argument == "--dry-run");
    let package_json = std::fs::read_to_string(&package_path).unwrap_or_else(|error| {
        eprintln!("Failed to read {}: {}", package_path.display(), error);
        std::process::exit(1);
    });
    match app_lib::commands::persona_memory::import_persona_memory_collection(
        user_id,
        package_json,
        Some(dry_run),
    )
    .await
    {
        Ok(result) => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
        Err(error) => {
            eprintln!("Import failed: {}", error);
            std::process::exit(1);
        }
    }
}
