fn main() {
  println!("cargo:rerun-if-changed=../../scripts/db/einstein_seed.sql");
  tauri_build::build()
}
