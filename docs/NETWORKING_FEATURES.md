# Zynkbot Networking Features (v0.9)

**Local-first, privacy-preserving networking across your devices**

Zynkbot's networking features enable device-to-device communication, memory synchronization, file sharing, and *potentially* distrubuted LLM inference—all without cloud dependency or third-party servers.

---

## Overview

Zynkbot provides three core networking capabilities:

1. **ZynkSync** - Memory synchronization across your devices
2. **ZynkLink** - File sharing between paired devices
3. **ZChat** - Direct device-to-device messaging

All features work over any shared local network (home WiFi, office LAN, or mobile hotspot) and maintain Zynkbot's privacy-first philosophy.  Developing these networking features was necessary to accomplish Zynkbots goals of maintaining a consistent experience across devices, the ability to function offline in inhospitable physical or political environments, and it enables the potential development of features such as Child mode - where a parent's Zynkbot might need to interact with a child's.

---

## 🔄 ZynkSync: Cross-Device Memory Synchronization

**Sync YOUR memories across YOUR devices (laptop, desktop, phone)**

### What It Does

- Automatically syncs conversation memories between paired devices
- Background synchronization every 60 seconds
- Complete data sync including embeddings, entities, and relationships

### Key Features

- ✅ **Automatic Sync**: Background sync at configurable intervals (default 60s)
- ✅ **Complete Data**: Syncs embeddings, entities, relationships, and metadata
- ✅ **Local Network Only**: Data never leaves your network

### Conflict Resolution (Two-Tier System)

Zynkbot uses two different conflict resolution strategies depending on the situation:

**1. Sync ID Conflicts (Automatic):**
When the same memory ID exists on both devices during sync:
- Backend automatically compares timestamps
- Newer memory overwrites older memory
- Prevents duplicate IDs in database
- Location: `zynksync.rs`

**2. Semantic/Content Conflicts (User Choice):**
When AI detects contradicting information during conversation:
- User presented with modal showing both memories
- Five resolution options:
  - Keep old memory (discard new)
  - Keep new memory (discard old)
  - Not a contradiction (dismiss without changes)
  - Keep both marked as contradictory
  - Keep both with explanation (creates a new explanation memory linked to both via `resolves` edges)
- Prevents hallucinations from AI guessing which is correct
- Location: `ConflictResolutionModal.jsx`

### Setup

1. Open **Settings → ZynkSync** on both devices
2. Note the IP address on Device 1 (port 57963)
3. On Device 2, click **"Add Device"** and enter Device 1's IP
4. Enter the 6-digit pairing code shown on Device 1
5. Sync starts automatically in the background

### Use Cases

**Multi-Device Personal Use:**
- Conversation history available on any device
- Seamless continuity across laptop, desktop, phone
- Offline-capable over WiFi/LAN
- All memories sync automatically

### Implementation

- **Backend**: Pure Rust async implementation (`zynkbot_rust/src-tauri/src/zynksync.rs`)
- **Protocol**: HTTP-based sync over port 57963
- **Storage**: Local SQLite database
- **Security**: 6-digit pairing codes, 10-minute timeout
- **Database**: All memories with `is_syncable = true` are synced (enforced in every sync query); `namespace` is preserved per memory but not yet used to filter what syncs — per-namespace sync control is planned

---

## 📁 ZynkLink: Device-to-Device File Sharing

**Share files and directories between paired Zynkbot devices**

### What It Does

- Share local folders with other paired devices
- Browse and download files from remote devices
- **Download files directly into your Zynkbot's knowledge base**
- No cloud storage required

### Key Features

- ✅ **Share Directories**: Make local folders accessible to other devices
- ✅ **Browse Files**: See files from paired devices in real-time
- ✅ **Download to Custom Location**: Pull files from remote devices to any local path
- ✅ **Download to Knowledge Base**: Instantly integrate shared documents into your AI's context
- ✅ **Read-only by default**: All shares are created read-only; write operations are not implemented
- ✅ **Security**: Path traversal protection, SHA256 file hashing

### Knowledge Base Integration

**Key Feature:** When browsing files from paired devices, you can download them directly to your Zynkbot's knowledge base with one click. This means:

- Shared documents instantly become part of your AI's context
- Your Zynkbot can answer questions about shared files immediately
- No manual import process - click "Download to KB" button
- Files are indexed and embedded automatically
- Great for team collaboration, family document sharing, or syncing research papers

**UI Location:** ZynkLinkPanel shows two download buttons:
- 📥 Download to Knowledge Base (automatic indexing)
- 📂 Download to Custom Location (manual save)

---

## You Don't Need a Router

Zynkbot networking requires only a shared local network — and a mobile hotspot counts. If one person enables a hotspot on their phone or laptop, any device that joins it is on the same local network and can pair with Zynkbot immediately.

**What this means in practice:**

- Two people in a location with no WiFi infrastructure can share files, sync memories, or message each other — one enables a hotspot, the other joins it, and they pair normally
- Field workers, journalists, aid workers, or researchers can link devices anywhere — no router, no internet, no IT department required
- Sharing a large local AI model (`.gguf` files are 2–8GB) between two laptops: hotspot, pair, ZynkLink transfer, done — nothing uploaded anywhere
- People caught in post-disaster or remote environments where internet is unavailable but local wireless still works

**Pairing over a hotspot is identical to pairing over a home network:**

1. Both devices connect to the same hotspot
2. Device A opens Settings → Zynklink and notes its IP address (shown in the UI)
3. Device B clicks "Add Device", enters Device A's IP and the 6-digit pairing code
4. Paired — file sharing and ZChat work immediately

The 6-digit code (10-minute expiry) handles security so no additional setup is needed.

**Current platform support:** Desktop and laptop (Windows and Linux). Android support is planned — once available, two phones can pair over a hotspot with no other hardware required.


### Use Cases

- Share documents between laptop and desktop
- Access work files from home computer
- Family photo/document sharing across devices
- Team collaboration: share research papers, code docs, reference materials
- Instant knowledge base updates from shared files

### Setup

1. **On sharing device**: Settings → ZynkLink → "Share Directory"
2. Select folder to share and set permissions
3. **On receiving device**: Browse shared directories from paired devices
4. Download files to custom location OR directly to knowledge base

### Security

- **Path Traversal Protection**: Validates all file paths to prevent `../` attacks
- **Read-Only Default**: Directories shared as read-only by default
- **Manifest Requirement**: Only scanned files are accessible
- **Pairing Required**: File sharing requires established ZynkLink pairing

### Implementation

- **Backend**: Pure Rust implementation (`zynkbot_rust/src-tauri/src/zynklink.rs`)
- **Frontend**: `ZynkLinkPanel.jsx` with KB integration
- **Protocol**: HTTP-based file serving
- **Storage**: Local SQLite tracking of shared directories
- **Database**: `zynk_linked_directories` and `zynk_file_manifest` tables

---

## 💬 ZChat: Device-to-Device Messaging

**Direct messaging between paired Zynkbot devices without cloud storage**

### What It Does

- Send text messages directly to any paired device
- Real-time delivery tracking and read receipts
- Voice input support via Web Speech API (optional)
- Local storage on device (not cloud)

### Key Features

- ✅ **Direct Messages**: Text messaging to any paired device
- ✅ **Delivery Tracking**: See when messages are delivered and read
- ✅ **Voice Input**: Optional voice dictation for message composition
- ✅ **Emoji Support**: Standard emoji picker included
- ✅ **Real-Time**: Messages appear instantly when devices are online (3-second polling)
- ✅ **Persistent**: Messages saved locally on device

### UI Features

**ZChatModal Component:**
- Opens per device from ZynkLink panel
- Scrollable message history
- Device name in modal title
- "You" vs remote device attribution
- Voice dictation button (🎤)
- Emoji picker (😊)
- Auto-scroll to bottom on new messages

### Use Cases

- Coordinate between family members' devices
- Team communication without Slack/Discord
- Private notes between your own devices
- Secure messaging for sensitive communications

### Message Delivery

**Sending:**
1. User types message or uses voice input
2. Message stored in local database
3. Immediate link attempt to target device
4. Delivery confirmation when received

**Receiving:**
- Frontend polls every 3 seconds when chat is open
- Background sync delivers messages instantly
- Messages marked as read when chat is opened

### Database Schema

```sql
CREATE TABLE zchat_messages (
    id UUID PRIMARY KEY,
    from_device_id UUID NOT NULL,
    to_device_id UUID NOT NULL,
    message_text TEXT NOT NULL,
    sent_at TIMESTAMP NOT NULL,
    delivered_at TIMESTAMP,
    read_at TIMESTAMP,
    user_id UUID NOT NULL
);
```

### Implementation

- **Backend**: Pure Rust async messaging (`zynkbot_rust/src-tauri/src/zchat.rs`)
- **Frontend**: React component (`zynkbot_rust/src/components/ZChatModal.jsx`)
- **Protocol**: Direct HTTP delivery via ZynkLink connection (port 57963)
- **Storage**: Local SQLite (`zchat_messages` table)

---

## Network Architecture

All networking features share common infrastructure:

### Device Pairing

- **6-digit pairing codes** (10-minute expiration)
- **Device registry** in local database (`zynk_devices` table)
- **Pairing relationships** (`zynk_device_pairings` table)
- **IP discovery** via user input

### Communication

- **Protocol**: HTTP-based communication
- **Port**: 57963 (ZynkSync/ZynkLink/ZChat)
- **Network**: Local network only (WiFi/LAN)
- **Security**: Pairing required, path validation

### Storage

- **Database**: Local SQLite for all metadata
- **Sync State**: Tracks what's been synced (`zynk_sync_state` table)
- **Messages**: Local storage only (`zchat_messages` table)
- **Files**: Local filesystem + manifest in database

---

## Security Considerations

### Current (v0.9 Production)

- ✅ **TLS 1.3 encryption** for all sync traffic (self-signed certs, automatic trust on pairing)
- ⚠️ **Pairing codes** expire after 10 minutes
- ⚠️ **Local network only** - not exposed to internet
- ✅ **Path validation** prevents directory traversal
- ✅ **Pairing required** for all network features

**Safe for:**
- Home networks
- Trusted local networks
- Private office networks

**NOT safe for:**
- Public WiFi
- Internet exposure
- Untrusted networks

### Future Enhancements

See [ROADMAP.md](ROADMAP.md) for planned security features:
- Device authentication with cryptographic keys
- Audit logs for all network operations
- Optional end-to-end encryption
- Integrity verification for synced data

---

## Troubleshooting

### ZynkSync Issues

**Devices can't connect:**
1. Verify both devices on same network
2. Check firewall allows port 57963
3. Confirm pairing codes match
4. Test ping between devices

**Commands to allow port 57963:**

**Windows:**
```batch
netsh advfirewall firewall add rule name="Zynkbot ZynkSync" dir=in action=allow protocol=TCP localport=57963
```

**Linux (UFW):**
```bash
sudo ufw allow 57963/tcp
```

**Linux (firewalld):**
```bash
sudo firewall-cmd --add-port=57963/tcp --permanent
sudo firewall-cmd --reload
```

### ZChat Not Delivering

**Possible causes:**
- Devices not paired
- Backend not running on remote device
- Network connectivity issue
- Firewall blocking port 57963

**Solution:**
1. Check device pairing status in Settings → ZynkLink
2. Verify backend running on both devices
3. Manually trigger sync from Settings

### Ensemble Mode Slow

**Possible causes:**
- Too many models selected (parallelizes, but coordinator waits for all)
- Slow API models (GPT-4 can take 5-10 seconds)
- Web search timeout (5 seconds)

**Solution:**
- Use fewer models (2-3 recommended)
- Mix fast local models with API models
- Check internet connection for web search

---

## Use Case Examples

### Example 1: Multi-Device Personal Use

**Setup:**
- Laptop for work
- Desktop at home
- Both running Zynkbot, paired via ZynkSync

**Configuration:**
- All memories sync automatically
- ZChat for notes between devices
- ZynkLink for file access

**Result:**
- Conversation history available on both devices
- Seamless continuity between work and home
- Offline-capable over home WiFi

### Example 2: Family File Sharing & Coordination

**Setup:**
- Parent's laptop
- Teen's desktop
- Both running Zynkbot, paired via ZynkLink

**What they use it for:**
- Teen shares homework documents with the parent's device via ZynkLink
- Parent downloads shared files directly into their Zynkbot's knowledge base to help review
- ZChat for quick coordination — works even if internet or cell service is temporarily out

**Result:**
- Teen shares a homework assignment; parent's Zynkbot can answer questions about it immediately
- Messages between devices go over the local network — no internet required
- Each person's memories remain private to their own device

### Example 3: Research Workflow

**Setup:**
- Researcher with laptop and desktop
- Multiple AI models available (local + API)

**Configuration:**
- ZynkSync keeps research notes synchronized
- ZynkLink for sharing datasets between devices
- Ensemble mode for fact-checking research findings

**Result:**
- Research notes on all devices
- Large datasets accessible via ZynkLink
- Multi-model verification prevents research errors
- Fact-checking with citations

---

## Platform Support

**Current (Rust/Tauri Desktop v0.9):**
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Arch, Fedora)
- ⚠️ macOS (not tested)

**Future:**
- 📱 Android (Tauri Mobile - primary platform goal)
- 📱 iOS (Tauri Mobile - planned)

---

## Performance

### ZynkSync

- **Sync interval**: 60 seconds (configurable)
- **Typical sync time**: Fast on a local network; varies by batch size and hardware
- **Network bandwidth**: ~10-50KB per memory (includes embeddings)

### ZynkLink

- **File transfer**: Depends on file size and network speed
- **Typical LAN speed**: 10-100 MB/s (WiFi 5/6)
- **No size limits**: Limited only by disk space
- **KB download**: Automatic indexing and embedding after download

### ZChat

- **Message latency**: 1-3 seconds (polling interval)
- **Bandwidth**: Negligible (~1KB per message)
- **Storage**: Unlimited (local SQLite)

---

## Related Documentation

- [README](../README.md) - Setup instructions
- [Database Schema](architecture_and_development/DATABASE_SCHEMA.md) - Database structure
- [Roadmap](ROADMAP.md) - Future networking features
- [Digital Resilience](DIGITAL_RESILIENCE.md) - Offline-first architecture
- [Project Vision](PROJECT_VISION.md) - ContainAI's mission

---

## License

Zynkbot networking features are part of Zynkbot and licensed under:
- **AGPL v3** - Free for non-commercial use
- **Commercial License** - Required for commercial use

See [LICENSE](../LICENSE) for full terms.
