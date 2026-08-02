use std::path::Path;
use std::fs;
use std::sync::Arc;
use sha2::{Sha256, Digest};
use tokio_rustls::rustls;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, DistinguishedName, Error as TlsError, SignatureScheme};

const CERT_PEM_FILE: &str = "zynkbot_tls_cert.pem";
const KEY_PEM_FILE:  &str = "zynkbot_tls_key.pem";
const CERT_DER_FILE: &str = "zynkbot_tls_cert.der";

/// Newtype carrying the raw DER bytes of the client's TLS certificate, injected
/// as a request extension by the accept loop when a client presents one.
#[derive(Clone, Debug)]
pub struct PeerCertDer(pub Vec<u8>);

/// Injected by verify_device_middleware when the peer's cert matches a paired device in the DB.
#[derive(Clone, Debug)]
pub struct VerifiedDevice {
    pub device_id: String,
    pub device_name: String,
}

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
    let _ = rustls::crypto::ring::default_provider().install_default();

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

/// Build a rustls ServerConfig that requests (but does not require) a client certificate.
/// The accept loop extracts the presented cert and injects it as a `PeerCertDer` extension
/// so middleware can verify device identity against the DB.
pub fn build_server_config_with_optional_client_auth(
    cert_pem: &str,
    key_pem: &str,
) -> Result<rustls::ServerConfig, String> {
    use rustls_pemfile::{certs, private_key};
    use std::io::BufReader;
    let _ = rustls::crypto::ring::default_provider().install_default();

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

    let verifier = Arc::new(OptionalClientCertVerifier);
    rustls::ServerConfig::builder()
        .with_client_cert_verifier(verifier)
        .with_single_cert(cert_chain, key)
        .map_err(|e| format!("Failed to build TLS ServerConfig with optional client auth: {}", e))
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

/// Custom rustls ServerCertVerifier that authenticates peers by exact DER byte comparison.
/// This replaces CA-chain verification: we pinned the peer's cert during pairing, so we
/// compare presented cert bytes against the pinned set instead of building a trust chain.
#[derive(Debug)]
pub struct PinnedCertVerifier {
    pub pinned_cert_ders: Vec<Vec<u8>>,
}

impl ServerCertVerifier for PinnedCertVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        let cert_der = end_entity.as_ref();
        if self.pinned_cert_ders.iter().any(|p| p.as_slice() == cert_der) {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(TlsError::General(format!(
                "Cert not pinned ({} bytes presented, {} pinned cert(s))",
                cert_der.len(),
                self.pinned_cert_ders.len()
            )))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// ClientCertVerifier that accepts any presented certificate without rejecting it.
/// We capture the cert in the accept loop and verify device identity against the DB
/// in middleware — the TLS layer just needs to not reject the cert at handshake time.
/// Setting client_auth_mandatory to false allows legacy clients (no cert) to connect.
#[derive(Debug)]
pub struct OptionalClientCertVerifier;

impl ClientCertVerifier for OptionalClientCertVerifier {
    fn offer_client_auth(&self) -> bool {
        true
    }

    fn client_auth_mandatory(&self) -> bool {
        false
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, TlsError> {
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Build a reqwest ClientConfig that uses pinned-cert verification.
/// Call this from rebuild_http_client instead of add_root_certificate.
pub fn build_pinned_client_config(pinned_ders: Vec<Vec<u8>>) -> rustls::ClientConfig {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let verifier = Arc::new(PinnedCertVerifier { pinned_cert_ders: pinned_ders });
    rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_no_client_auth()
}

/// Build a reqwest ClientConfig with pinned-cert verification AND a client certificate.
/// This presents our own device cert during the TLS handshake so the server can verify
/// our identity by cert rather than trusting the x-device-id header alone.
pub fn build_pinned_client_config_with_cert(
    pinned_ders: Vec<Vec<u8>>,
    cert_pem: &str,
    key_pem: &str,
) -> Result<rustls::ClientConfig, String> {
    use rustls_pemfile::{certs, private_key};
    use std::io::BufReader;
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_chain: Vec<rustls::pki_types::CertificateDer<'static>> = {
        let mut reader = BufReader::new(cert_pem.as_bytes());
        certs(&mut reader)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to parse client cert PEM: {}", e))?
    };

    let key = {
        let mut reader = BufReader::new(key_pem.as_bytes());
        private_key(&mut reader)
            .map_err(|e| format!("Failed to parse client key PEM: {}", e))?
            .ok_or_else(|| "No private key found in client key PEM".to_string())?
    };

    let verifier = Arc::new(PinnedCertVerifier { pinned_cert_ders: pinned_ders });
    rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(cert_chain, key)
        .map_err(|e| format!("Failed to build TLS ClientConfig with client cert: {}", e))
}
