# 🏗️ Snap-in Architecture Guide

**Document Purpose:** Technical implementation guide for building Zynkbot Snap-ins

**Audience:** External developers, contributors, and maintainers

**Related Documents:**
- [Professional Snap-ins](./professional_snap_ins.md) - Use cases and vision
- [Personal Snap-ins](./personal_snap_ins.md) - Personal use case catalog

---

## 📍 Current Implementation Status

### Proof-of-Concept: Therapist Snap-in

Zynkbot currently includes **one hardcoded Snap-in example** (Therapist Journal) that demonstrates the core concepts but is not yet a full plugin system.  What is required for a full plugin system is understood and architecturally it is straightforward to build pending interest by developers.  Currently on the roadmap.

**Current Architecture:**
- **Frontend:** `zynkbot_rust/src/components/SnapInModal.jsx` (368 lines, hardcoded UI)
- **Backend:** `index_snapin_notes()` Tauri command in `src-tauri/src/lib.rs`
- **Storage:** File path namespacing (`snap_ins/therapist/{patient}/{session}.txt`)
- **Indexing:** Integrates with existing RAG system using namespace isolation

**What Works:**
- ✅ File path isolation prevents cross-contamination between snap-in data
- ✅ RAG integration with privacy boundaries
- ✅ Mode-aware consent checks
- ✅ Local-first storage (no external dependencies)

**What's Missing:**
- ❌ Plugin manifest system
- ❌ Dynamic snap-in loading
- ❌ Snap-in discovery/registry
- ❌ Developer SDK/API
- ❌ Standardized UI hooks
- ❌ Snap-in lifecycle management (install/uninstall/update)

---

## 🎯 Target Architecture: Full Plugin System

### Design Principles

1. **Local-First:** Snap-ins should run without external dependencies when possible
2. **Consent-Driven:** All data access must go through consent layer
3. **Mode-Aware:** Snap-ins respect containment mode boundaries (Guardian, Child, Sovereign, Witness, HIPAA)
4. **Namespace Isolated:** Each snap-in has its own storage namespace
5. **Fail-Safe:** Snap-in failures never crash the main application
6. **Auditable:** All snap-in actions are logged for user review

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Zynkbot Core                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Snap-in Runtime Manager                     │   │
│  │  - Discovery & Registration                         │   │
│  │  - Lifecycle Management (install/enable/disable)    │   │
│  │  - Permission Enforcement                           │   │
│  │  - Error Isolation                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Snap-in API Layer                           │   │
│  │  - Storage API (namespaced)                         │   │
│  │  - RAG API (semantic search)                        │   │
│  │  - UI Hook API (modal, sidebar, input extensions)   │   │
│  │  - Mode API (check current mode)                    │   │
│  │  - Consent API (request user permission)            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │
         ┌────────────────┴───────────────┐
         │                                │
    ┌────▼─────┐                   ┌─────▼────┐
    │ Snap-in  │                   │ Snap-in  │
    │    A     │                   │    B     │
    │          │                   │          │
    │ manifest │                   │ manifest │
    │ UI comp  │                   │ UI comp  │
    │ backend  │                   │ backend  │
    └──────────┘                   └──────────┘
```

---

## 📋 Snap-in Manifest Format

Each snap-in must provide a `snapin.json` manifest in its root directory:

```json
{
  "id": "therapist_journal",
  "name": "Therapist Journal",
  "version": "1.0.0",
  "author": "Zynkbot Team",
  "description": "Private session note management for therapists",

  "permissions": [
    "storage.write",
    "storage.read",
    "rag.index",
    "rag.search",
    "ui.modal"
  ],

  "modes": ["hipaa", "guardian"],

  "entry_points": {
    "ui_component": "./src/TherapistModal.jsx",
    "backend_command": "./src/commands.rs",
    "icon": "🧠"
  },

  "storage": {
    "namespace": "snap_ins/therapist",
    "file_patterns": ["**/*.txt", "**/*.md"]
  },

  "ui_hooks": {
    "sidebar_button": {
      "label": "Therapist Notes",
      "icon": "🧠",
      "position": "bottom-left"
    },
    "settings_panel": true
  },

  "consent_requirements": {
    "on_install": "This snap-in will store patient session notes locally on your device.",
    "on_first_use": "Indexing notes allows semantic search across sessions.",
    "data_scope": "Patient names, session titles, and note content"
  },

  "dependencies": {
    "zynkbot_sdk": "^1.0.0"
  }
}
```

### Manifest Field Definitions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | ✅ | Unique identifier (snake_case) |
| `name` | ✅ | Human-readable name |
| `version` | ✅ | Semantic version (x.y.z) |
| `author` | ✅ | Developer name or organization |
| `description` | ✅ | Brief description (max 200 chars) |
| `permissions` | ✅ | Array of required permissions |
| `modes` | ✅ | Compatible modes: `["guardian", "child", "sovereign", "witness", "hipaa"]` |
| `entry_points` | ✅ | File paths to UI and backend code |
| `storage.namespace` | ✅ | Storage path prefix (must start with `snap_ins/`) |
| `ui_hooks` | ⚠️ | Optional UI integration points |
| `consent_requirements` | ✅ | User-facing consent prompts |
| `dependencies` | ⚠️ | External dependencies |

---

## 🔌 Snap-in SDK/API

### Storage API

```rust
// Rust backend API
use zynkbot_sdk::storage::SnapinStorage;

#[tauri::command]
async fn my_snapin_save(
    content: String,
    file_path: String,
    user_id: String,
    snapin_id: String
) -> Result<(), String> {
    // SDK automatically prepends namespace: snap_ins/{snapin_id}/{file_path}
    SnapinStorage::write(snapin_id, file_path, content, user_id).await?;
    Ok(())
}

#[tauri::command]
async fn my_snapin_read(
    file_path: String,
    user_id: String,
    snapin_id: String
) -> Result<String, String> {
    SnapinStorage::read(snapin_id, file_path, user_id).await
}
```

```javascript
// Frontend API (React)
import { snapinStorage } from '@zynkbot/sdk';

const saveNote = async (content, path) => {
  await snapinStorage.write({
    snapinId: 'therapist_journal',
    filePath: path,
    content: content,
    userId: currentUserId
  });
};
```

### RAG API

```rust
// Index content for semantic search
use zynkbot_sdk::rag::SnapinRAG;

#[tauri::command]
async fn my_snapin_index(
    content: String,
    file_path: String,
    metadata: HashMap<String, String>,
    user_id: String,
    snapin_id: String
) -> Result<(), String> {
    SnapinRAG::index(
        snapin_id,
        file_path,
        content,
        metadata,
        user_id
    ).await?;
    Ok(())
}

#[tauri::command]
async fn my_snapin_search(
    query: String,
    user_id: String,
    snapin_id: String,
    limit: usize
) -> Result<Vec<SearchResult>, String> {
    // Automatically filters to snapin's namespace
    SnapinRAG::search(snapin_id, query, user_id, limit).await
}
```

### Mode API

```javascript
// Check current mode before showing sensitive UI
import { modeAPI } from '@zynkbot/sdk';

const MySnapinModal = () => {
  const currentMode = modeAPI.getCurrentMode(); // "guardian" | "child" | "sovereign" | "witness" | "hipaa"

  // Example: restrict a HIPAA snap-in to HIPAA mode only
  if (currentMode !== "hipaa") {
    return <div>This snap-in requires HIPAA mode. Switch modes in Settings.</div>;
  }

  return <MySnapinContent />;
};
```

### Consent API

```javascript
// Request user consent before sensitive operations
import { consentAPI } from '@zynkbot/sdk';

const indexNotes = async () => {
  const granted = await consentAPI.request({
    action: "index_patient_notes",
    scope: "Patient names and session content will be processed for semantic search",
    dataRetention: "Indexed until manually deleted",
    snapinId: "therapist_journal"
  });

  if (!granted) {
    console.log("User declined consent");
    return;
  }

  // Proceed with indexing...
};
```

---

## 🏗️ File Structure Template

```
my_snapin/
├── snapin.json                 # Manifest (required)
├── README.md                   # Documentation
├── LICENSE                     # License file
│
├── src/
│   ├── components/
│   │   └── MySnapinModal.jsx   # React UI component
│   │
│   ├── commands/
│   │   └── mod.rs              # Rust Tauri commands
│   │
│   └── styles/
│       └── MySnapin.css        # Component styles
│
└── tests/
    ├── integration.test.js
    └── unit.test.rs
```

---

## 🛠️ Example: Minimal Snap-in Implementation

### 1. Create Manifest (`snapin.json`)

```json
{
  "id": "simple_note",
  "name": "Simple Note Taker",
  "version": "1.0.0",
  "author": "Your Name",
  "description": "Basic note-taking with semantic search",

  "permissions": [
    "storage.write",
    "storage.read",
    "rag.index",
    "rag.search",
    "ui.modal"
  ],

  "modes": ["sovereign", "guardian"],

  "entry_points": {
    "ui_component": "./src/SimpleNoteModal.jsx",
    "backend_command": "./src/commands.rs",
    "icon": "📝"
  },

  "storage": {
    "namespace": "snap_ins/simple_note"
  },

  "ui_hooks": {
    "sidebar_button": {
      "label": "Quick Notes",
      "icon": "📝",
      "position": "bottom-left"
    }
  },

  "consent_requirements": {
    "on_install": "This snap-in stores notes locally on your device.",
    "data_scope": "Note titles and content"
  }
}
```

### 2. Create UI Component (`src/SimpleNoteModal.jsx`)

```jsx
import React, { useState } from 'react';
import { snapinStorage, snapinRAG } from '@zynkbot/sdk';

export default function SimpleNoteModal({ isOpen, onClose, userId, snapinId }) {
  const [noteTitle, setNoteTitle] = useState('');
  const [noteContent, setNoteContent] = useState('');

  const saveNote = async () => {
    if (!noteTitle.trim() || !noteContent.trim()) {
      alert('Please enter both title and content');
      return;
    }

    try {
      // Save file
      const filePath = `${noteTitle.trim()}.txt`;
      await snapinStorage.write({
        snapinId,
        filePath,
        content: noteContent,
        userId
      });

      // Index for search
      await snapinRAG.index({
        snapinId,
        filePath,
        content: noteContent,
        metadata: { title: noteTitle },
        userId
      });

      alert('Note saved!');
      setNoteTitle('');
      setNoteContent('');
    } catch (error) {
      alert(`Failed to save note: ${error}`);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <h2>Quick Note</h2>

        <input
          type="text"
          value={noteTitle}
          onChange={(e) => setNoteTitle(e.target.value)}
          placeholder="Note title..."
        />

        <textarea
          value={noteContent}
          onChange={(e) => setNoteContent(e.target.value)}
          placeholder="Note content..."
          rows={10}
        />

        <button onClick={saveNote}>Save Note</button>
        <button onClick={onClose}>Cancel</button>
      </div>
    </div>
  );
}
```

### 3. Create Backend Commands (`src/commands.rs`)

```rust
use zynkbot_sdk::{storage::SnapinStorage, rag::SnapinRAG};
use tauri::command;
use std::collections::HashMap;

#[command]
pub async fn simple_note_save(
    title: String,
    content: String,
    user_id: String,
) -> Result<String, String> {
    let snapin_id = "simple_note".to_string();
    let file_path = format!("{}.txt", title.trim());

    // Save file
    SnapinStorage::write(snapin_id.clone(), file_path.clone(), content.clone(), user_id.clone()).await?;

    // Index for search
    let mut metadata = HashMap::new();
    metadata.insert("title".to_string(), title);

    SnapinRAG::index(
        snapin_id,
        file_path,
        content,
        metadata,
        user_id
    ).await?;

    Ok("Note saved successfully".to_string())
}

#[command]
pub async fn simple_note_search(
    query: String,
    user_id: String,
    limit: usize
) -> Result<Vec<SearchResult>, String> {
    let snapin_id = "simple_note".to_string();
    SnapinRAG::search(snapin_id, query, user_id, limit).await
}
```

### 4. Register Snap-in (Auto-discovery in future, manual for now)

```rust
// In main.rs or lib.rs
use my_snapin::commands::{simple_note_save, simple_note_search};

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // ... existing commands ...
            simple_note_save,
            simple_note_search,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 🔐 Permission System

### Available Permissions

| Permission | Description | Risk Level |
|------------|-------------|------------|
| `storage.read` | Read files in snap-in namespace | Low |
| `storage.write` | Write files in snap-in namespace | Low |
| `storage.read_global` | Read files outside namespace | High |
| `rag.index` | Add content to vector database | Low |
| `rag.search` | Search within snap-in namespace | Low |
| `rag.search_global` | Search across all namespaces | Medium |
| `ui.modal` | Display modal dialogs | Low |
| `ui.sidebar` | Add sidebar buttons | Low |
| `ui.input_extension` | Modify message input area | Medium |
| `mode.read` | Check current mode | Low |
| `network.request` | Make external API calls | High |
| `memory.read` | Access chat memory | High |
| `memory.write` | Modify chat memory | Critical |

### Permission Enforcement

Permissions are enforced at the SDK layer. Unauthorized API calls throw errors:

```rust
// Example: Snap-in tries to access file outside namespace
SnapinStorage::read("other_snapin", "file.txt", user_id).await?;
// Error: "Permission denied: storage.read_global required"
```

---

## 🔄 Snap-in Lifecycle

### Installation Flow

1. User places snap-in folder in `~/.zynkbot/snap_ins/` (or uses install command)
2. Zynkbot validates `snapin.json` manifest
3. Checks for permission conflicts or mode incompatibilities
4. Shows consent prompt with permissions and data scope
5. User accepts/declines
6. If accepted:
   - Registers snap-in in local database
   - Creates namespace directory
   - Loads UI components
   - Registers backend commands
7. Snap-in appears in UI (sidebar button, settings panel, etc.)

### Update Flow

1. User downloads new version
2. Zynkbot compares manifests
3. Shows changelog and new permissions (if any)
4. User accepts/declines
5. If accepted:
   - Backs up current version
   - Replaces files
   - Migrates data if migration script provided
   - Reloads snap-in

### Uninstall Flow

1. User clicks "Uninstall Snap-in" in settings
2. Zynkbot shows data deletion options:
   - Delete all data (files + indexed content)
   - Keep data for later reinstall
   - Export data before deletion
3. User confirms
4. Removes snap-in registration
5. Optionally deletes namespace directory
6. Removes UI hooks

---

## 🧪 Testing Requirements

Every snap-in should include:

### Unit Tests (Backend)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_save_note() {
        let result = simple_note_save(
            "Test".to_string(),
            "Content".to_string(),
            "user123".to_string()
        ).await;

        assert!(result.is_ok());
    }
}
```

### Integration Tests (Frontend)
```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import SimpleNoteModal from '../SimpleNoteModal';

test('saves note on button click', async () => {
  render(<SimpleNoteModal isOpen={true} userId="test" />);

  const titleInput = screen.getByPlaceholderText('Note title...');
  const contentInput = screen.getByPlaceholderText('Note content...');
  const saveButton = screen.getByText('Save Note');

  fireEvent.change(titleInput, { target: { value: 'Test Note' } });
  fireEvent.change(contentInput, { target: { value: 'Test Content' } });
  fireEvent.click(saveButton);

  // Assert success
});
```

### Consent Flow Tests
```javascript
test('shows consent prompt on first use', async () => {
  // Mock consent API
  const mockConsent = jest.fn(() => Promise.resolve(true));
  consentAPI.request = mockConsent;

  // Trigger action requiring consent
  await saveNote();

  expect(mockConsent).toHaveBeenCalledWith({
    action: "index_patient_notes",
    scope: expect.any(String),
    snapinId: "therapist_journal"
  });
});
```

---

## 🚀 Roadmap: From POC to Full Plugin System

### Phase 1: Foundation
- Extract hardcoded therapist snap-in into separate module
- Create `zynkbot-sdk` crate with Storage API
- Implement manifest validation
- Create snap-in registry (SQLite table)
- Add namespace enforcement to file operations

### Phase 2: Dynamic Loading
- Auto-discovery of snap-ins in `~/.zynkbot/snap_ins/`
- Dynamic Tauri command registration
- React component lazy loading
- Permission enforcement layer
- Consent API implementation

### Phase 3: Developer Experience
- CLI tool: `zynkbot-cli create-snapin`
- Snap-in project templates
- Hot reload for snap-in development
- Developer documentation site
- Example snap-ins repository

### Phase 4: Distribution
- Snap-in marketplace (optional, privacy-preserving)
- Code signing and verification
- Automated testing framework
- Migration tools for updates
- Community contribution guidelines

---

## 🛡️ Security Considerations

### Sandboxing
- Each snap-in runs in its own namespace
- No direct access to core Zynkbot memory
- API calls are rate-limited to prevent abuse
- Snap-in crashes are isolated (don't crash main app)

### Code Review
- All snap-ins in official repository undergo security review
- Third-party snap-ins show warning on install
- Users can inspect snap-in code before installation

### Data Privacy
- Snap-ins cannot access other snap-ins' data without explicit permission
- All network requests logged and shown to user
- Consent required before indexing personal data
- Users can audit all snap-in data at any time

---

## 📚 Resources for Developers

### Documentation
- [Zynkbot SDK Reference](./SDK_REFERENCE.md) *(coming soon)*
- [Containment Modes](../FEATURES.md#-containment-modes)
- [Memory Processing Pipeline](../architecture_and_development/MEMORY_PROCESSING_PIPELINE.md)

### Example Snap-ins
- Therapist Journal (current POC)
- Simple Note Taker (minimal example above)
- Code Companion *(coming soon)*
- Parenting Companion *(planned)*

### Community
- GitHub Discussions: Feature requests and Q&A
- Issue Tracker: Bug reports and enhancements
- Contact: matt@containai.ai

---

## 🤝 Contributing

To contribute a snap-in:

1. Fork the repository
2. Create snap-in using template above
3. Add comprehensive tests
4. Document all features and permissions
5. Submit PR with:
   - Snap-in code
   - Tests
   - README
   - Case study in `/docs/case_studies/` (if applicable)

All snap-ins must:
- Pass automated security checks
- Include consent documentation
- Respect mode boundaries

---

## 📝 Notes

**Current State:** This document describes the target architecture. As of May 2026, Zynkbot has a **proof-of-concept** with one hardcoded snap-in. The SDK and plugin system are under development.

**Vision vs. Reality:** The vision documents ([professional_snap_ins.md](./professional_snap_ins.md), [personal_snap_ins.md](./personal_snap_ins.md)) describe the long-term goal. This document bridges the gap between vision and implementation.

---

**Document Version:** 1.0.0
**Last Updated:** April 2026
**Contact:** matt@containai.ai
