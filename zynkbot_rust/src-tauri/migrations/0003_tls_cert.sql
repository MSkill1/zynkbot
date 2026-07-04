-- Add TLS certificate storage to device table.
-- Each paired peer's cert DER is stored here and pinned for all future connections.
ALTER TABLE zynk_devices ADD COLUMN tls_cert_der BLOB;
