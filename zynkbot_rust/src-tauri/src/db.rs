use sqlx::{sqlite::SqlitePoolOptions, SqlitePool};
use std::path::PathBuf;

pub fn get_app_data_dir() -> PathBuf {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("zynkbot");
    std::fs::create_dir_all(&data_dir).ok();
    data_dir
}

pub fn get_models_dir() -> PathBuf {
    let data_models = get_app_data_dir().join("models");
    if data_models.exists() {
        return data_models;
    }
    // Dev mode fallback: models live in src-tauri/models/ relative to the exe in target/
    if let Ok(exe) = std::env::current_exe() {
        if exe.to_string_lossy().contains("target") {
            if let Some(exe_dir) = exe.parent() {
                let dev_models = exe_dir.parent()
                    .and_then(|p| p.parent())
                    .unwrap_or(exe_dir)
                    .join("models");
                if dev_models.exists() {
                    return dev_models;
                }
            }
        }
    }
    data_models
}

pub fn get_db_path() -> PathBuf {
    get_app_data_dir().join("zynkbot.db")
}

pub fn get_user_profile_path() -> PathBuf {
    get_app_data_dir().join("user_profile.json")
}

pub fn get_db_url() -> String {
    format!("sqlite://{}?mode=rwc", get_db_path().display())
}

pub async fn create_pool() -> Result<SqlitePool, sqlx::Error> {
    let url = get_db_url();
    println!("[Rust DB] Opening SQLite database: {}", url);

    // SQLite PRAGMAs that are per-connection must be set via after_connect so every
    // connection checked out of the pool gets them. Setting them only on the pool
    // object (as was done before) applies only to one connection; other connections
    // in the pool silently inherit SQLite defaults (foreign_keys=OFF, etc.).
    let pool = SqlitePoolOptions::new()
        .max_connections(20)
        .acquire_timeout(std::time::Duration::from_secs(5))
        .after_connect(|conn, _meta| Box::pin(async move {
            sqlx::query("PRAGMA journal_mode=WAL").execute(&mut *conn).await?;
            sqlx::query("PRAGMA foreign_keys=ON").execute(&mut *conn).await?;
            sqlx::query("PRAGMA synchronous=NORMAL").execute(&mut *conn).await?;
            sqlx::query("PRAGMA busy_timeout=15000").execute(&mut *conn).await?;
            Ok(())
        }))
        .connect(&url)
        .await?;

    // Run migrations — idempotent, sqlx tracks applied versions in _sqlx_migrations
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .map_err(|e| sqlx::Error::Configuration(format!("Migration failed: {}", e).into()))?;

    println!("[Rust DB] ✅ Connected to SQLite (schema up to date)");
    Ok(pool)
}

#[allow(dead_code)]
pub async fn test_connection(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let result: (i32,) = sqlx::query_as("SELECT 1").fetch_one(pool).await?;
    println!("[Rust DB] Test query result: {}", result.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .connect("sqlite::memory:")
            .await
            .expect("in-memory DB failed");
        sqlx::query("PRAGMA foreign_keys=ON").execute(&pool).await.unwrap();
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("migration failed");
        pool
    }

    #[tokio::test]
    async fn migration_creates_all_tables() {
        let pool = test_pool().await;
        let expected = [
            "memories", "memory_links",
            "kb_documents", "kb_chunks",
            "zynk_devices", "zynk_device_pairings", "zynk_sync_state",
            "user_sync_codes",
            "zynk_linked_directories", "zynk_file_manifest", "zynk_link_manifest",
            "zynklink_codes", "zynklink_pairings",
            "zchat_messages",
            "conversation_sessions", "conversation_messages", "message_feedback",
            "zynk_device_certificates",
        ];
        for table in &expected {
            let exists: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?"
            )
            .bind(table)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert_eq!(exists.0, 1, "missing table: {}", table);
        }
    }

    #[tokio::test]
    async fn memory_relationships_view_exists() {
        let pool = test_pool().await;
        let (count,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name='memory_relationships'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn can_insert_and_query_memory() {
        let pool = test_pool().await;
        sqlx::query(
            "INSERT INTO memories (content, namespace) VALUES (?, ?)"
        )
        .bind("Test memory content")
        .bind("personal")
        .execute(&pool)
        .await
        .expect("insert failed");

        let (content,): (String,) = sqlx::query_as(
            "SELECT content FROM memories WHERE namespace = ?"
        )
        .bind("personal")
        .fetch_one(&pool)
        .await
        .expect("select failed");

        assert_eq!(content, "Test memory content");
    }

    #[tokio::test]
    async fn fts5_table_exists() {
        let pool = test_pool().await;
        let (count,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='conversation_messages_fts'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }
}
