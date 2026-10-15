//
//  ChatterboxText.swift
//  MLXAudio
//
//  Chatterbox text preprocessing (Multilingual V3).
//  Ported from mlx-audio Python: chatterbox/chatterbox.py (punc_norm) and
//  chatterbox/tokenizer.py (MTLTokenizer, ChineseCangjieConverter).
//

import Foundation
import Tokenizers

/// Language codes accepted by the multilingual checkpoints.
let chatterboxSupportedLanguages: Set<String> = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
]

// MARK: - Punctuation normalization

/// Python `str.isspace` also counts U+001C...U+001F, which Unicode White_Space does not.
private func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
    scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
}

/// Exact port of Python `punc_norm`. Operates on scalars, not Characters, to match Python's code-point semantics.
func chatterboxPuncNorm(_ text: String, capitalize: Bool) -> String {
    if text.isEmpty {
        return "You need to add some text for me to talk."
    }

    var text = text
    if capitalize, let first = text.unicodeScalars.first, first.properties.isLowercase {
        text = String(first).uppercased() + String(text.unicodeScalars.dropFirst())
    }

    // Collapse whitespace runs (and trim) like `" ".join(text.split())`.
    var words: [String] = []
    var current = String.UnicodeScalarView()
    for scalar in text.unicodeScalars {
        if isPythonWhitespace(scalar) {
            if !current.isEmpty {
                words.append(String(current))
                current = String.UnicodeScalarView()
            }
        } else {
            current.append(scalar)
        }
    }
    if !current.isEmpty { words.append(String(current)) }
    text = words.joined(separator: " ")

    // The Python quote entries are no-ops in the shipped package (ASCII to ASCII), so they are omitted.
    let replacements: [(String, String)] = [
        ("...", ", "), ("\u{2026}", ", "), (":", ","), (" - ", ", "), (";", ", "),
        ("\u{2014}", "-"), ("\u{2013}", "-"), (" ,", ","),
    ]
    for (old, new) in replacements {
        text = text.replacingOccurrences(of: old, with: new, options: .literal)
    }

    var scalars = Array(text.unicodeScalars)
    while scalars.last == " " { scalars.removeLast() }
    let enders: Set<Unicode.Scalar> = [".", "!", "?", "-", ",", "\u{3001}", "\u{FF0C}", "\u{3002}", "\u{FF1F}", "\u{FF01}"]
    if let last = scalars.last, enders.contains(last) {
        return String(String.UnicodeScalarView(scalars))
    }
    return String(String.UnicodeScalarView(scalars)) + "."
}

// MARK: - Chinese Cangjie conversion

/// Port of `ChineseCangjieConverter` without segmentation (Python skips it when pkuseg is absent).
struct ChatterboxCangjieConverter: Sendable {
    // Keyed by code points: String equality would fold canonically equivalent compatibility ideographs.
    private let wordToCode: [[UInt32]: String]
    private let codeToWords: [String: [[UInt32]]]

    static let fileName = "Cangjie5_TC.json"

    /// Loads `Cangjie5_TC.json` from `modelDir`; an unreadable table yields pass-through, like Python.
    init(modelDir: URL) {
        var wordToCode: [[UInt32]: String] = [:]
        var codeToWords: [String: [[UInt32]]] = [:]
        do {
            let data = try Data(contentsOf: modelDir.appendingPathComponent(Self.fileName))
            let entries = try JSONDecoder().decode([String].self, from: data)
            for entry in entries {
                let parts = entry.components(separatedBy: "\t")
                guard parts.count >= 2 else { break }  // Python's unpack raises here and stops loading
                let word = parts[0].unicodeScalars.map(\.value)
                wordToCode[word] = parts[1]
                codeToWords[parts[1], default: []].append(word)
            }
        } catch {
            print("Warning: Could not load Cangjie mapping: \(error)")
        }
        self.wordToCode = wordToCode
        self.codeToWords = codeToWords
    }

    private func encode(_ glyph: Unicode.Scalar) -> String? {
        let key = [glyph.value]
        guard let code = wordToCode[key] else { return nil }  // e.g. hiragana
        let index = codeToWords[code]?.firstIndex(of: key) ?? 0
        return index > 0 ? code + String(index) : code
    }

    func convert(_ text: String) -> String {
        var output = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            guard scalar.properties.generalCategory == .otherLetter, let code = encode(scalar) else {
                output.append(scalar)
                continue
            }
            for c in code { output.append(contentsOf: "[cj_\(c)]".unicodeScalars) }
            output.append(contentsOf: "[cj_.]".unicodeScalars)
        }
        return String(output)
    }
}

// MARK: - V3 text preparation

/// Builds the exact string fed to BPE for Multilingual V3 (before the model adds SOT/EOT ids).
/// Russian stress marking is skipped, as in Python when `russian_text_stresser` is absent.
func prepareV3Text(
    _ text: String, language: String, cangjie: ChatterboxCangjieConverter?
) -> String {
    let language = language.lowercased()
    var prepared = chatterboxPuncNorm(text, capitalize: false)
    if language == "zh", let cangjie {
        prepared = cangjie.convert(prepared)
    }
    prepared = prepared.decomposedStringWithCompatibilityMapping  // NFKD, no lowercasing
    return "[\(language)]" + prepared.replacingOccurrences(of: " ", with: "[SPACE]", options: .literal)
}

// MARK: - BPE encoding

/// Encodes like HF `tokenizers`: each out-of-vocab character becomes one `[UNK]`.
/// swift-transformers force-unwraps ids (crashes) and byte-expands unknowns when
/// tokenizer_config.json has no `unk_token`, which the model's generated config lacks.
func chatterboxEncode(_ text: String, tokenizer: any Tokenizer) -> [Int] {
    let unk = tokenizer.convertTokenToId("[UNK]")
    func inVocab(_ token: String) -> Bool {
        guard let id = tokenizer.convertTokenToId(token) else { return false }
        return id != unk
    }

    let scalars = Array(text.unicodeScalars)
    var rebuilt = String.UnicodeScalarView()
    var i = 0
    while i < scalars.count {
        let scalar = scalars[i]
        // Keep added tokens such as [SPACE], [en], [cj_o] intact.
        if scalar == "[", let close = scalars[i...].firstIndex(of: "]") {
            let candidate = String(String.UnicodeScalarView(scalars[i...close]))
            if inVocab(candidate) {
                rebuilt.append(contentsOf: candidate.unicodeScalars)
                i = close + 1
                continue
            }
        }
        // Whitespace is dropped by the pre-tokenizer, never mapped to [UNK].
        if scalar.properties.isWhitespace || inVocab(String(scalar)) {
            rebuilt.append(scalar)
        } else {
            rebuilt.append(contentsOf: "[UNK]".unicodeScalars)
        }
        i += 1
    }

    return tokenizer.tokenize(text: String(rebuilt)).compactMap { tokenizer.convertTokenToId($0) ?? unk }
}
