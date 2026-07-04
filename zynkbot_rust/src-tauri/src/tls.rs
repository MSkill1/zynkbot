use std::path::Path;
use std::fs;
use sha2::{Sha256, Digest};
use tokio_rustls::rustls;

const CERT_PEM_FILE: &str = "zynkbot_tls_cert.pem";
const KEY_PEM_FILE:  &str = "zynkbot_tls_key.pem";
const CERT_DER_FILE: &str = "zynkbot_tls_cert.der";

/// Load or generate this device's self-signed TLS certificate.
/// Returns (cert_pem, key_pem, cert_der).
/// Files are stored in data_dir and reused across restarts.
pub fn load_or_generate_cert(data_dir: &Path) -> Result<(String, String, Vec<u8>), String> {
    let cert_pem_path = data_dir.join(CERT_PEM_FILE);
    let key_pem_path  = data_dir.join(KEY_PEM_FILE);
    let cert_der_path = data_dir.join(CERT_DER_FILE);

    if cert_pem_path.exists() && key_pem_path.exists() && cert_der_path.exists() {
        let cert_pem = fs::read_to_string(&cert_pem_path)
            .map_err(|e| format!("Failed to read cert PEM: {}", e))?;
        let key_pem = fs::read_to_string(&key_pem_path)
            .map_err(|e| format!("Failed to read key PEM: {}", e))?;
        let cert_der = fs::read(&cert_der_path)
            .map_err(|e| format!("Failed to read cert DER: {}", e))?;

        println!("[TLS] Loaded existing certificate (fingerprint: {})", cert_fingerprint(&cert_der));
        return Ok((cert_pem, key_pem, cert_der));
    }

    generate_and_save(data_dir)
}

fn generate_and_save(data_dir: &Path) -> Result<(String, String, Vec<u8>), String> {
    use rcgen::{generate_simple_self_signed, CertifiedKey};

    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(vec!["localhost".to_string()])
            .map_err(|e| format!("Failed to generate TLS certificate: {}", e))?;

    let cert_pem = cert.pem();
    let key_pem  = key_pair.serialize_pem();
    let cert_der = cert.der().to_vec();

    fs::create_dir_all(data_dir)
        .map_err(|e| format!("Failed to create data dir: {}", e))?;
    fs::write(data_dir.join(CERT_PEM_FILE), &cert_pem)
        .map_err(|e| format!("Failed to write cert PEM: {}", e))?;
    fs::write(data_dir.join(KEY_PEM_FILE), &key_pem)
        .map_err(|e| format!("Failed to write key PEM: {}", e))?;
    fs::write(data_dir.join(CERT_DER_FILE), &cert_der)
        .map_err(|e| format!("Failed to write cert DER: {}", e))?;

    println!("[TLS] Generated new self-signed certificate (fingerprint: {})", cert_fingerprint(&cert_der));
    Ok((cert_pem, key_pem, cert_der))
}

/// Build a rustls ServerConfig from PEM-encoded cert and key.
/// Used by start_http_server() to create the TLS acceptor.
pub fn build_server_config(cert_pem: &str, key_pem: &str) -> Result<rustls::ServerConfig, String> {
    use rustls_pemfile::{certs, private_key};
    use std::io::BufReader;

    let cert_chain: Vec<rustls::pki_types::CertificateDer<'static>> = {
        let mut reader = BufReader::new(cert_pem.as_bytes());
        certs(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse cert PEM: {}", e))?
    };

    let key = {
        let mut reader = BufReader::new(key_pem.as_bytes());
        private_key(&mut reader)
            .map_err(|e| format!("Failed to parse key PEM: {}", e))?
            .ok_or_else(|| "No private key found in PEM".to_string())?
    };

    rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .map_err(|e| format!("Failed to build TLS ServerConfig: {}", e))
}

/// SHA-256 fingerprint of a cert DER — shown during pairing for user verification.
pub fn cert_fingerprint(der: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(der);
    hasher.finalize()
        .iter()
        .map(|b| format!("{:02X}", b))
        .collect::<Vec<_>>()
        .join(":")
}
