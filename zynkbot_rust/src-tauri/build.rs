fn main() {
  println!("cargo:rerun-if-changed=seeds/einstein_seed.sql");
  tauri_build::build()
}
