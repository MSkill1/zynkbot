-- Add sync_paired to distinguish ZynkSync pairings from ZynkLink pairings.
-- ZynkLink sets is_paired=1 but does not set sync_paired, so linked devices
-- no longer appear in the ZynkSync device list.
-- Existing devices with a TLS cert were paired via ZynkSync — backfill them.
ALTER TABLE zynk_devices ADD COLUMN sync_paired INTEGER NOT NULL DEFAULT 0;
UPDATE zynk_devices SET sync_paired = 1 WHERE tls_cert_der IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_zynk_devices_sync_paired ON zynk_devices(sync_paired);
