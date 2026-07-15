/// User Identity Management
///
/// Manages persistent user and device identities across devices:
/// - user_id: Identifies a USER across all their devices (for memory sync)
/// - device_id: Identifies a specific DEVICE (for file sharing)
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIdentity {
    pub user_id: String,
    pub device_id: String,
    pub user_id_file: String,
    pub device_id_file: String,
}

pub struct IdentityManager {
    base_path: PathBuf,
    user_id_file: PathBuf,
    device_id_file: PathBuf,
}

impl IdentityManager {
    /// Create a new identity manager with the specified base path
    pub fn new(base_path: Option<PathBuf>) -> Self {
        let base = base_path.unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });

        Self {
            user_id_file: base.join(".zynk_user_id"),
            device_id_file: base.join(".zynk_device_id"),
            base_path: base,
        }
    }

    /// Get or create user_id
    pub fn get_or_create_user_id(&self) -> Result<String, String> {
        if self.user_id_file.exists() {
            if let Ok(user_id) = fs::read_to_string(&self.user_id_file) {
                let trimmed = user_id.trim();
                if !trimmed.is_empty() {
                    return Ok(trimmed.to_string());
                }
            }
        }

        // Create new user identity
        let user_id = Uuid::new_v4().to_string();
        fs::write(&self.user_id_file, &user_id)
            .map_err(|e| format!("Failed to write user_id: {}", e))?;

        // Create info file
        self.create_info_file()?;

        Ok(user_id)
    }

    /// Get or create device_id
    pub fn get_or_create_device_id(&self) -> Result<String, String> {
        if self.device_id_file.exists() {
            if let Ok(device_id) = fs::read_to_string(&self.device_id_file) {
                let trimmed = device_id.trim();
                if !trimmed.is_empty() {
                    return Ok(trimmed.to_string());
                }
            }
        }

        // Create new device identity
        let device_id = Uuid::new_v4().to_string();
        fs::write(&self.device_id_file, &device_id)
            .map_err(|e| format!("Failed to write device_id: {}", e))?;

        Ok(device_id)
    }

    /// Set user_id manually (for device syncing via code)
    pub fn set_user_id(&self, user_id: &str) -> Result<(), String> {
        fs::write(&self.user_id_file, user_id)
            .map_err(|e| format!("Failed to write user_id: {}", e))?;
        Ok(())
    }

    /// Get complete identity information
    pub fn get_identity(&self) -> Result<UserIdentity, String> {
        Ok(UserIdentity {
            user_id: self.get_or_create_user_id()?,
            device_id: self.get_or_create_device_id()?,
            user_id_file: self.user_id_file.to_string_lossy().to_string(),
            device_id_file: self.device_id_file.to_string_lossy().to_string(),
        })
    }

    /// Reset device_id only (new device, same user)
    #[allow(dead_code)]
    pub fn reset_device_only(&self) -> Result<String, String> {
        let new_device_id = Uuid::new_v4().to_string();
        fs::write(&self.device_id_file, &new_device_id)
            .map_err(|e| format!("Failed to write device_id: {}", e))?;
        Ok(new_device_id)
    }

    /// Reset both user_id and device_id (completely new identity)
    #[allow(dead_code)]
    pub fn reset_all(&self) -> Result<(String, String), String> {
        let new_user_id = Uuid::new_v4().to_string();
        let new_device_id = Uuid::new_v4().to_string();

        fs::write(&self.user_id_file, &new_user_id)
            .map_err(|e| format!("Failed to write user_id: {}", e))?;
        fs::write(&self.device_id_file, &new_device_id)
            .map_err(|e| format!("Failed to write device_id: {}", e))?;

        Ok((new_user_id, new_device_id))
    }

    /// Create explanatory info file
    fn create_info_file(&self) -> Result<(), String> {
        let info_file = self.base_path.join(".zynk_user_info.txt");
        if info_file.exists() {
            return Ok(());
        }

        let info_text = "\
ZYNKBOT USER IDENTITY
====================

Your user_id (in .zynk_user_id) identifies YOU across all your devices.
- Same user_id = Your devices sync memories automatically
- Different user_id = Different person, can share files via ZynkLink

To link a new device to your account:
1. Copy .zynk_user_id from one of your devices
2. Paste it to the new device
3. Restart Zynkbot

To share with another person:
- Let them keep their own .zynk_user_id
- Use ZynkLink to share specific folders
";

        fs::write(&info_file, info_text)
            .map_err(|e| format!("Failed to write info file: {}", e))?;

        Ok(())
    }
}

/// Global identity manager instance
static IDENTITY_MANAGER: once_cell::sync::Lazy<IdentityManager> =
    once_cell::sync::Lazy::new(|| {
        // On Android, dirs::config_dir() returns a path outside the writable files/
        // directory, causing every launch to generate a new random UUID fallback.
        // Use get_app_data_dir() instead, which resolves to $HOME/files/zynkbot.
        #[cfg(target_os = "android")]
        let identity_base = crate::db::get_app_data_dir();

        #[cfg(not(target_os = "android"))]
        let identity_base = {
            let config_dir = dirs::config_dir()
                .unwrap_or_else(|| {
                    eprintln!("[Identity] WARNING: Could not get config dir, using home");
                    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
                })
                .join("Zynkbot");
            if !config_dir.exists() {
                let _ = std::fs::create_dir_all(&config_dir);
            }
            config_dir
        };

        println!("[Identity] Looking for user identity files in: {:?}", identity_base);
        IdentityManager::new(Some(identity_base))
    });

/// Get or create user_id
pub fn get_user_id() -> Result<String, String> {
    IDENTITY_MANAGER.get_or_create_user_id()
}

/// Get or create device_id
pub fn get_device_id() -> Result<String, String> {
    IDENTITY_MANAGER.get_or_create_device_id()
}

/// Get complete identity
pub fn get_identity() -> Result<UserIdentity, String> {
    IDENTITY_MANAGER.get_identity()
}

/// Set user_id manually
pub fn set_user_id(user_id: &str) -> Result<(), String> {
    IDENTITY_MANAGER.set_user_id(user_id)
}

/// Reset both user_id and device_id (completely new identity)
pub fn reset_all_identity() -> Result<(String, String), String> {
    IDENTITY_MANAGER.reset_all()
}
