//! A seal the user makes (Matt, 2026-10-27).
//!
//! One chosen thread, exported as a signed file. The file carries the thread's text, a
//! SHA-256 digest of it, and an Ed25519 signature over the digest and the sealing time,
//! made with a key that lives only on this device. Anyone holding the file can check
//! that the text has not changed since the seal was made and that the same key made it.
//!
//! What it does not do, on purpose: nothing is written into the database. There is no
//! chain over every message, nothing that certifies the record against its owner when
//! the device is taken. Evidence is made by the user, when they choose, about what they
//! choose. The sealing time is this device's clock; a timestamp from a third party, only
//! when the user asks for one, is what would prove the date to anyone else.
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const FORMAT: &str = "zynkbot-seal/1";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SealedMessage {
    pub role: String,
    pub at: String,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SealedThread {
    pub session_id: String,
    pub title: String,
    pub started_at: String,
    pub messages: Vec<SealedMessage>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SealFile {
    pub format: String,
    /// When the seal was made, by this device's clock (RFC 3339, UTC).
    pub sealed_at: String,
    /// The device's public key, hex. Shown to the user so they can tell their own seals.
    pub signer: String,
    pub thread: SealedThread,
    /// SHA-256 of the thread, hex.
    pub digest: String,
    /// Ed25519 over the format, the sealing time and the digest, hex.
    pub signature: String,
}

/// The digest covers the thread exactly as serialised: serde_json writes struct fields in
/// declaration order, so the bytes are the same on every platform and every run.
pub fn thread_digest(thread: &SealedThread) -> String {
    let bytes = serde_json::to_vec(thread).expect("a plain struct serialises");
    hex::encode(Sha256::digest(&bytes))
}

fn signed_bytes(format: &str, sealed_at: &str, digest: &str) -> Vec<u8> {
    format!("{}\n{}\n{}\n", format, sealed_at, digest).into_bytes()
}

pub fn seal(thread: SealedThread, key: &SigningKey, sealed_at: &str) -> SealFile {
    let digest = thread_digest(&thread);
    let signature = key.sign(&signed_bytes(FORMAT, sealed_at, &digest));
    SealFile {
        format: FORMAT.to_string(),
        sealed_at: sealed_at.to_string(),
        signer: hex::encode(key.verifying_key().to_bytes()),
        thread,
        digest,
        signature: hex::encode(signature.to_bytes()),
    }
}

#[derive(Serialize, Debug)]
pub struct Verdict {
    pub ok: bool,
    pub reason: String,
    pub sealed_at: String,
    pub signer: String,
    pub title: String,
    pub messages: usize,
}

pub fn verify(file: &SealFile) -> Verdict {
    let mut v = Verdict {
        ok: false,
        reason: String::new(),
        sealed_at: file.sealed_at.clone(),
        signer: file.signer.clone(),
        title: file.thread.title.clone(),
        messages: file.thread.messages.len(),
    };
    if file.format != FORMAT {
        v.reason = format!("unknown seal format \"{}\"", file.format);
        return v;
    }
    if thread_digest(&file.thread) != file.digest {
        v.reason = "the conversation text does not match the seal: it was changed after sealing".to_string();
        return v;
    }
    let key = hex::decode(&file.signer).ok()
        .and_then(|b| <[u8; 32]>::try_from(b).ok())
        .and_then(|b| VerifyingKey::from_bytes(&b).ok());
    let Some(key) = key else {
        v.reason = "the signer key is not readable".to_string();
        return v;
    };
    let Some(sig) = hex::decode(&file.signature).ok().and_then(|b| Signature::from_slice(&b).ok()) else {
        v.reason = "the signature is not readable".to_string();
        return v;
    };
    match key.verify(&signed_bytes(&file.format, &file.sealed_at, &file.digest), &sig) {
        Ok(()) => {
            v.ok = true;
            v.reason = "intact: the text matches the seal, and the seal was made by the signer shown".to_string();
        }
        Err(_) => {
            v.reason = "the signature does not match: the sealing time or the digest was changed, or another key made it".to_string();
        }
    }
    v
}

/// The device's signing key: a 32-byte seed in a file only this user can read. Made on
/// the first seal and kept; a new key would make every earlier seal read as "another key".
pub fn load_or_create_key(path: &Path) -> Result<SigningKey, String> {
    if let Ok(bytes) = std::fs::read(path) {
        let seed: [u8; 32] = bytes.as_slice().try_into()
            .map_err(|_| format!("{} is not a 32-byte key", path.display()))?;
        return Ok(SigningKey::from_bytes(&seed));
    }
    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {}", dir.display(), e))?;
    }
    std::fs::write(path, key.to_bytes()).map_err(|e| format!("cannot write {}: {}", path.display(), e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(key)
}

pub fn key_path() -> PathBuf {
    crate::db::get_app_data_dir().join("keys").join("seal.key")
}

/// Where seals are written: the Zynkbot folder in the user's Documents on desktop; on a
/// phone the ZynkbotShare folder, which the Files app shows as "Zynkbot".
pub fn seals_dir() -> PathBuf {
    #[cfg(target_os = "android")]
    {
        let data = crate::db::get_app_data_dir();
        data.parent().map(|p| p.join("ZynkbotShare")).unwrap_or(data).join("seals")
    }
    #[cfg(not(target_os = "android"))]
    {
        dirs::document_dir().unwrap_or_else(crate::db::get_app_data_dir).join("Zynkbot").join("seals")
    }
}

/// `<title>-<date>.zynkseal.json`, with the title reduced to letters, digits and dashes.
pub fn file_name(title: &str, sealed_at: &str) -> String {
    let mut slug: String = title.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-");
    slug.truncate(40);
    if slug.is_empty() { slug = "thread".to_string(); }
    let date: String = sealed_at.chars().take(19).map(|c| if c == ':' { '-' } else { c }).collect();
    format!("{}-{}.zynkseal.json", slug, date)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread() -> SealedThread {
        SealedThread {
            session_id: "s1".into(),
            title: "Parking at the mall".into(),
            started_at: "2026-10-26T18:00:00+00:00".into(),
            messages: vec![
                SealedMessage { role: "user".into(), at: "2026-10-26T18:00:01+00:00".into(), text: "I parked on level 3 at the mall".into() },
                SealedMessage { role: "assistant".into(), at: "2026-10-26T18:00:04+00:00".into(), text: "Noted: level 3.".into() },
            ],
        }
    }

    #[test]
    fn a_fresh_seal_verifies_and_names_its_signer() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        let file = seal(thread(), &key, "2026-10-27T12:00:00+00:00");
        let v = verify(&file);
        assert!(v.ok, "{}", v.reason);
        assert_eq!(v.signer, hex::encode(key.verifying_key().to_bytes()));
        assert_eq!(v.messages, 2);
        // The file survives a trip through JSON unchanged.
        let back: SealFile = serde_json::from_str(&serde_json::to_string_pretty(&file).unwrap()).unwrap();
        assert!(verify(&back).ok);
    }

    #[test]
    fn changed_text_is_caught_by_the_digest() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        let mut file = seal(thread(), &key, "2026-10-27T12:00:00+00:00");
        file.thread.messages[0].text = "I parked on level 4 at the mall".into();
        let v = verify(&file);
        assert!(!v.ok);
        assert!(v.reason.contains("changed after sealing"), "{}", v.reason);
    }

    #[test]
    fn a_changed_sealing_time_or_another_key_is_caught_by_the_signature() {
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        let mut file = seal(thread(), &key, "2026-10-27T12:00:00+00:00");
        file.sealed_at = "2026-09-01T12:00:00+00:00".into();
        let v = verify(&file);
        assert!(!v.ok && v.reason.contains("signature does not match"), "{}", v.reason);

        // Re-signing with another key makes a valid file, but the signer shown is not ours.
        let other = SigningKey::generate(&mut rand::rngs::OsRng);
        let forged = seal(thread(), &other, "2026-10-27T12:00:00+00:00");
        let v = verify(&forged);
        assert!(v.ok);
        assert_ne!(v.signer, hex::encode(key.verifying_key().to_bytes()));
    }

    #[test]
    fn the_key_is_made_once_and_kept() {
        let dir = std::env::temp_dir().join(format!("zynk-seal-{}", std::process::id()));
        let path = dir.join("keys").join("seal.key");
        let a = load_or_create_key(&path).unwrap();
        let b = load_or_create_key(&path).unwrap();
        assert_eq!(a.to_bytes(), b.to_bytes());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_names_are_plain() {
        assert_eq!(file_name("Parking at the mall!", "2026-10-27T12:00:00+00:00"), "parking-at-the-mall-2026-10-27T12-00-00.zynkseal.json");
        assert_eq!(file_name("", "2026-10-27T12:00:00+00:00"), "thread-2026-10-27T12-00-00.zynkseal.json");
    }
}
