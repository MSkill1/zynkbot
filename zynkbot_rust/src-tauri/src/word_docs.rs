//! Word documents for the Knowledge Base and attachments (Matt's father, 2026-10-27).
//!
//! `.docx` is a zip; the text is in `word/document.xml` as `<w:t>` runs inside `<w:p>`
//! paragraphs. `.doc` (Word 97-2003) is an OLE compound file: the `WordDocument` stream
//! holds the characters and a piece table in the `0Table`/`1Table` stream says where each
//! run of text sits and whether it is 8-bit (Windows-1252) or UTF-16. Only the words are
//! taken; no formatting, pictures, headers or footnotes.
use std::io::Read;

/// The paragraphs of a `.docx`, one per line.
pub fn docx_text(bytes: &[u8]) -> Result<String, String> {
    let cursor = std::io::Cursor::new(bytes);
    let mut zip = zip::ZipArchive::new(cursor).map_err(|e| format!("not a .docx (zip) file: {}", e))?;
    let mut xml = String::new();
    zip.by_name("word/document.xml")
        .map_err(|_| "not a Word document: no word/document.xml inside".to_string())?
        .read_to_string(&mut xml)
        .map_err(|e| format!("cannot read word/document.xml: {}", e))?;
    Ok(document_xml_text(&xml))
}

/// The text of WordprocessingML: `<w:t>` runs joined, a newline at each paragraph end,
/// a tab for `<w:tab/>`, a newline for `<w:br/>`; everything else dropped.
pub fn document_xml_text(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    let mut in_text = false;
    while let Some(lt) = rest.find('<') {
        if in_text {
            out.push_str(&unescape(&rest[..lt]));
        }
        rest = &rest[lt + 1..];
        let Some(gt) = rest.find('>') else { break };
        let tag = &rest[..gt];
        rest = &rest[gt + 1..];
        let name = tag.trim_start_matches('/').split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        let closing = tag.starts_with('/');
        let empty = tag.ends_with('/');
        match name {
            "w:t" => in_text = !closing && !empty,
            "w:p" if closing => out.push('\n'),
            "w:tab" if !closing => out.push('\t'),
            "w:br" | "w:cr" if !closing => out.push('\n'),
            "w:tc" if closing => out.push('\t'),
            _ => {}
        }
    }
    tidy(out)
}

fn unescape(s: &str) -> String {
    if !s.contains('&') { return s.to_string(); }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let Some(semi) = rest.find(';') else { out.push_str(rest); return out; };
        let entity = &rest[1..semi];
        let replacement = match entity {
            "amp" => Some('&'), "lt" => Some('<'), "gt" => Some('>'), "quot" => Some('"'), "apos" => Some('\''),
            _ if entity.starts_with("#x") => u32::from_str_radix(&entity[2..], 16).ok().and_then(char::from_u32),
            _ if entity.starts_with('#') => entity[1..].parse::<u32>().ok().and_then(char::from_u32),
            _ => None,
        };
        match replacement {
            Some(c) => { out.push(c); rest = &rest[semi + 1..]; }
            None => { out.push('&'); rest = &rest[1..]; }
        }
    }
    out.push_str(rest);
    out
}

/// Collapse runs of blank lines to one and trim the ends.
fn tidy(s: String) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank = 0;
    for line in s.lines() {
        let t = line.trim_end();
        if t.trim().is_empty() {
            blank += 1;
            if blank == 1 { out.push('\n'); }
        } else {
            blank = 0;
            out.push_str(t);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

fn u16_at(b: &[u8], i: usize) -> Option<u16> { b.get(i..i + 2).map(|s| u16::from_le_bytes([s[0], s[1]])) }
fn u32_at(b: &[u8], i: usize) -> Option<u32> { b.get(i..i + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])) }

/// The main text of a `.doc` (Word 97-2003), one paragraph per line.
pub fn doc_text(bytes: &[u8]) -> Result<String, String> {
    let cursor = std::io::Cursor::new(bytes);
    let mut cfb = cfb::CompoundFile::open(cursor).map_err(|e| format!("not a .doc (OLE) file: {}", e))?;
    let mut word = Vec::new();
    cfb.open_stream("/WordDocument")
        .map_err(|_| "not a Word document: no WordDocument stream".to_string())?
        .read_to_end(&mut word).map_err(|e| format!("cannot read WordDocument: {}", e))?;
    if u16_at(&word, 0) != Some(0xA5EC) {
        return Err("not a Word document: wrong signature".to_string());
    }
    let flags = u16_at(&word, 0x0A).ok_or("file too short")?;
    let table_name = if flags & 0x0200 != 0 { "/1Table" } else { "/0Table" };
    let ccp_text = u32_at(&word, 0x4C).ok_or("file too short")? as usize;
    let fc_clx = u32_at(&word, 0x1A2).ok_or("file too short")? as usize;
    let lcb_clx = u32_at(&word, 0x1A6).ok_or("file too short")? as usize;
    let mut table = Vec::new();
    cfb.open_stream(table_name)
        .map_err(|_| format!("not a Word document: no {} stream", &table_name[1..]))?
        .read_to_end(&mut table).map_err(|e| format!("cannot read the table stream: {}", e))?;
    let clx = table.get(fc_clx..fc_clx + lcb_clx).ok_or("the piece table is outside the table stream")?;

    // Clx: zero or more Prc (0x01, u16 size, bytes), then Pcdt (0x02, u32 size, PlcPcd).
    let mut i = 0;
    while clx.get(i) == Some(&0x01) {
        let n = u16_at(clx, i + 1).ok_or("bad piece table")? as usize;
        i += 3 + n;
    }
    if clx.get(i) != Some(&0x02) {
        return Err("the piece table is not where the file says".to_string());
    }
    let lcb = u32_at(clx, i + 1).ok_or("bad piece table")? as usize;
    let plc = clx.get(i + 5..i + 5 + lcb).ok_or("bad piece table")?;
    let pieces = (lcb.saturating_sub(4)) / 12;
    let mut text = String::new();
    let mut remaining = ccp_text;
    for p in 0..pieces {
        if remaining == 0 { break; }
        let cp_start = u32_at(plc, p * 4).ok_or("bad piece")? as usize;
        let cp_end = u32_at(plc, (p + 1) * 4).ok_or("bad piece")? as usize;
        let pcd = (pieces + 1) * 4 + p * 8;
        let fc_raw = u32_at(plc, pcd + 2).ok_or("bad piece")?;
        let compressed = fc_raw & 0x4000_0000 != 0;
        let fc = (fc_raw & 0x3FFF_FFFF) as usize;
        let chars = (cp_end.saturating_sub(cp_start)).min(remaining);
        remaining -= chars;
        if compressed {
            let start = fc / 2;
            let run = word.get(start..start + chars).ok_or("piece outside the document")?;
            text.extend(run.iter().map(|&b| cp1252(b)));
        } else {
            let run = word.get(fc..fc + chars * 2).ok_or("piece outside the document")?;
            let units: Vec<u16> = run.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            text.push_str(&String::from_utf16_lossy(&units));
        }
    }
    Ok(tidy(clean_doc_chars(&text)))
}

/// Word's own control characters: paragraph marks become newlines, cell marks tabs,
/// field codes and the rest are dropped.
fn clean_doc_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_field_code = false;
    for c in s.chars() {
        match c {
            '\r' | '\u{000B}' | '\u{000C}' => out.push('\n'),
            '\u{0007}' => out.push('\t'),
            '\u{0013}' => in_field_code = true,       // field begin: the code follows
            '\u{0014}' => in_field_code = false,      // field separator: the result follows
            '\u{0015}' => in_field_code = false,      // field end
            '\u{0001}' | '\u{0008}' => {}             // picture / drawn object anchors
            c if (c as u32) < 0x20 && c != '\t' && c != '\n' => {}
            c => if !in_field_code { out.push(c) },
        }
    }
    out
}

/// Windows-1252 to Unicode for the 8-bit pieces; the 0x80–0x9F block differs from Latin-1.
fn cp1252(b: u8) -> char {
    const HIGH: [u16; 32] = [
        0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x008D, 0x017D, 0x008F,
        0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
    ];
    match b {
        0x80..=0x9F => char::from_u32(HIGH[(b - 0x80) as usize] as u32).unwrap_or('?'),
        _ => b as char,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOCX: &[u8] = include_bytes!("../tests/fixtures/soap_notes.docx");
    const DOC: &[u8] = include_bytes!("../tests/fixtures/soap_notes.doc");

    fn expect_soap_notes(text: &str) {
        for line in ["Soap making notes", "Melt 500 g of coconut oil. Add the lye solution slowly & stir until trace.",
                     "Pour into the mould; cure for four weeks.", "Second heading", "The bars are ready when they no longer feel soft."] {
            assert!(text.contains(line), "missing {:?} in:\n{}", line, text);
        }
        assert!(!text.contains('<') && !text.contains("w:"), "markup leaked:\n{}", text);
    }

    #[test]
    fn docx_paragraphs_come_out_as_lines() {
        let text = docx_text(DOCX).unwrap();
        expect_soap_notes(&text);
        assert_eq!(text.lines().filter(|l| !l.trim().is_empty()).count(), 5, "{}", text);
    }

    #[test]
    fn doc_paragraphs_come_out_as_lines() {
        let text = doc_text(DOC).unwrap();
        expect_soap_notes(&text);
    }

    #[test]
    fn xml_runs_tabs_breaks_and_entities() {
        let xml = r#"<w:document><w:body><w:p><w:r><w:t>Tom &amp; Jerry</w:t></w:r><w:r><w:tab/><w:t xml:space="preserve"> 2 &lt; 3</w:t></w:r></w:p><w:p><w:r><w:t>second</w:t><w:br/><w:t>line &#233;</w:t></w:r></w:p></w:body></w:document>"#;
        assert_eq!(document_xml_text(xml), "Tom & Jerry\t 2 < 3\nsecond\nline é");
    }

    #[test]
    fn the_wrong_kind_of_file_is_refused_plainly() {
        assert!(docx_text(b"not a zip").unwrap_err().contains("not a .docx"));
        assert!(doc_text(b"not ole").unwrap_err().contains("not a .doc"));
        assert!(docx_text(DOC).unwrap_err().contains("not a .docx"));
        assert!(doc_text(DOCX).unwrap_err().contains("not a .doc"));
    }
}
