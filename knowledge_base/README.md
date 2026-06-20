# Zynkbot Knowledge Base

The Knowledge Base lets you give Zynkbot access to your own documents — reports, notes, manuals, research papers, code, or anything else you want it to be able to read and reference.

This is separate from Zynkbot's personal memory. Memory is for things Zynkbot learns about *you* over time. The Knowledge Base is for reference material you deliberately load in — things you want to look up, discuss, or surface in conversation.

---

## How to add and index documents

1. Open the side panel and go to **Settings → Knowledge Base**
2. Click **Open Folder** to open your KB folder in the file manager
3. Copy or move your files into that folder
4. Click **Knowledge Base Manager** to open the document manager
5. Your files appear under **Available Files** — they are not yet searchable
6. Click **Index** next to each file you want to index, or **Index All** to index everything at once

Indexing is a manual step. Files in your KB folder are not searchable until you index them. This lets you keep files in the folder without making all of them part of your search index. When Linked to another Zynkbot, you can download shared files to a folder on your device, or if they are in a supported file format, directly into your Zynkbot's knowledge base for automatic indexing and immediate availability to your own Zynkbot.

---

## Managing indexed documents

Inside the Knowledge Base Manager:

- **Indexed Documents** — files that have been indexed and are available for search. Each shows its file size, number of chunks, and when it was last indexed.
- **Available Files** — files in your KB folder that have not been indexed yet.
- 🔄 **Re-index** — regenerates all embeddings for a document. Use this if the file has changed.
- 🗑️ **Remove** — removes the document from the search index. The file itself is not deleted from your KB folder.

---

## Supported file types

| Category | Extensions |
|---|---|
| Text and docs | `.txt` `.md` `.csv` `.json` `.log` |
| Code | `.rs` `.js` `.jsx` `.ts` `.tsx` `.py` `.java` `.cpp` `.c` `.h` |
| Markup and config | `.html` `.css` `.xml` `.yaml` `.yml` `.toml` |

**Coming soon:** `.pdf` `.docx` and other common document formats.

---

## How indexing works

When you index a file, Zynkbot:

1. Splits the document into overlapping chunks of roughly 500 characters, breaking at sentence boundaries where possible
2. Generates a semantic embedding for each chunk using the local `all-MiniLM-L6-v2` model — this runs entirely on your device, nothing is sent anywhere
3. Stores the chunks and their embeddings in your local database

When you search, Zynkbot embeds your query the same way and finds the chunks whose meaning is closest to what you asked.

---

## Searching your Knowledge Base

KB search does not run on every message — you turn it on when you need it.

**To search:** Click the **📚 KB** button in the chat input bar before sending your message. Zynkbot searches your indexed documents and brings the most relevant sections into the conversation.

Once those results are in the conversation, Zynkbot can discuss them with you across multiple turns without searching again. This is intentional — it is more efficient than re-searching the database on every message, and it gives you control over when KB content enters the conversation rather than having it silently influence every reply. If you want fresh results, click the **📚 KB** button again on your next message.

If you mention a filename in your message (e.g., "what does requirements.txt say about..."), Zynkbot recognizes the reference and prioritizes results from that file.

---

## Downloading files from a linked Zynkbot

If you have ZynkLink set up with another Zynkbot device — a colleague, a family member, or your own second machine — you can browse and download files from their shared folders. Each file in the listing has two buttons:

- **→ KB** — downloads the file into your Knowledge Base folder and automatically indexes it, making it immediately available for search
- **Save...** — opens your file manager so you can save the file anywhere on your system; the file is not added to the Knowledge Base

The other device must be online and running Zynkbot for either option to work.

---

## How files are stored

Files that you want Zynkbot to be able to search must be in the Knowledge Base folder inside this project directory:

```
knowledge_base/
├── {your-user-id}/         # Your indexed documents live here
│   ├── document.txt
│   └── ...
└── README.md
```

This is the only location Zynkbot scans when you open the Knowledge Base Manager. Files you download via **Save...** in ZynkLink go wherever you choose on your file system — they are not in the Knowledge Base unless you copy them into this folder and index them manually.

Some snap-ins store documents in their own subdirectories — for example, the Therapist snap-in organizes patient session files separately. Those are excluded from general KB searches and are only accessible within the snap-in that manages them.

---

## Testing that it works

A sample document is automatically copied into your KB folder the first time you launch Zynkbot: `sample_knowledge_base_document.txt`. It covers six obscure historical topics — the Antikythera Mechanism, the Wow! Signal, the Voynich Manuscript, and others — with suggested test questions at the end.

To test:
1. Open **Knowledge Base Manager** and index the sample document
2. Ask one of the test questions from the bottom of the file with the **📚 KB** button on
3. If Zynkbot returns a specific answer from the document rather than general knowledge, indexing and retrieval are working correctly

---

## Troubleshooting

**File doesn't appear in Available Files:** Check that the file extension is in the supported list above. Unsupported file types are not shown. Click **Refresh** to re-scan the folder after adding files.

**Answers seem wrong or generic:** Make sure you clicked the 📚 KB button before sending, and that the relevant document has been indexed (not just placed in the folder).

**ZynkLink download fails:** The other device must be online, running Zynkbot, and reachable on your local network. Check that they have shared the folder containing the file.

**Indexing is slow:** Large files take longer because the embedding model runs locally on CPU. A progress bar is shown during indexing.
