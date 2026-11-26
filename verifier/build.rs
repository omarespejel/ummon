use std::path::Path;
use std::process::Command;

fn resolve_git_sha() -> String {
    if let Ok(val) = std::env::var("UMMON_GIT_SHA") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(out) = Command::new("git").args(["rev-parse", "HEAD"]).output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "unknown".to_string()
}

fn resolve_git_dirty() -> &'static str {
    if let Ok(val) = std::env::var("UMMON_GIT_DIRTY") {
        let low = val.trim().to_ascii_lowercase();
        if matches!(low.as_str(), "1" | "true" | "yes") {
            return "true";
        }
        if matches!(low.as_str(), "0" | "false" | "no") {
            return "false";
        }
    }
    if let Ok(out) = Command::new("git")
        .args([
            "-C",
            "..",
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let any_dirty = text.lines().any(|line| {
                let t = line.trim();
                !t.is_empty() && t != "?? fixtures/" && t != "?? fixtures"
            });
            return if any_dirty { "true" } else { "false" };
        }
    }
    "true"
}

fn main() {
    println!("cargo:rerun-if-env-changed=UMMON_GIT_SHA");
    println!("cargo:rerun-if-env-changed=UMMON_GIT_DIRTY");
    if Path::new("../.git/HEAD").exists() {
        println!("cargo:rerun-if-changed=../.git/HEAD");
    }
    if Path::new("../.git/index").exists() {
        println!("cargo:rerun-if-changed=../.git/index");
    }
    println!("cargo:rustc-env=UMMON_BUILD_GIT_SHA={}", resolve_git_sha());
    println!(
        "cargo:rustc-env=UMMON_BUILD_GIT_DIRTY={}",
        resolve_git_dirty()
    );
}
