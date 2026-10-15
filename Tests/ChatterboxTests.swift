import Foundation
import MLX
import Testing
import Tokenizers

@testable import MLXAudioTTS

@Suite("Chatterbox generation tests")
struct ChatterboxGenerationTests {
    @Test func emotionOverrideAppliesToDefaultConditioning() {
        let defaults = MLXArray(Float(0.5))

        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: nil).item(Float.self) == 0.5)
        #expect(
            resolveChatterboxEmotionAdv(default: defaults, override: 0.9).item(Float.self) == 0.9)
    }
}

// MARK: - Multilingual V3 text parity

private struct V3TextFixture: Decodable {
    let language: String
    let text: String
    let puncNorm: String
    let prepared: String
    let ids: [Int]

    enum CodingKeys: String, CodingKey {
        case language, text, prepared, ids
        case puncNorm = "punc_norm"
    }
}

private struct V3TextFixtureFile: Decodable {
    let cases: [V3TextFixture]
}

private func loadV3Fixtures() throws -> [V3TextFixture] {
    let url = try #require(
        Bundle.module.url(
            forResource: "chatterbox_v3_text_fixtures", withExtension: "json",
            subdirectory: "media"))
    return try JSONDecoder().decode(V3TextFixtureFile.self, from: Data(contentsOf: url)).cases
}

/// Real V3 model folder (tokenizer.json, Cangjie5_TC.json); gated tests skip without it.
private let v3ModelDir: URL? = ProcessInfo.processInfo.environment["CHATTERBOX_V3_DIR"]
    .map { URL(fileURLWithPath: $0) }

@Suite("Chatterbox V3 text tests")
struct ChatterboxV3TextTests {
    @Test func puncNormMatchesPython() throws {
        for fixture in try loadV3Fixtures() {
            #expect(
                chatterboxPuncNorm(fixture.text, capitalize: false) == fixture.puncNorm,
                "\(fixture.language): \(fixture.text)")
        }
    }

    @Test func puncNormCapitalizesAndHandlesEmpty() {
        #expect(chatterboxPuncNorm("hello there", capitalize: true) == "Hello there.")
        #expect(chatterboxPuncNorm("hello there", capitalize: false) == "hello there.")
        #expect(chatterboxPuncNorm("", capitalize: false) == "You need to add some text for me to talk.")
    }

    @Test(.enabled(if: v3ModelDir != nil))
    func preparedTextMatchesPython() throws {
        let cangjie = ChatterboxCangjieConverter(modelDir: try #require(v3ModelDir))
        for fixture in try loadV3Fixtures() {
            let prepared = prepareV3Text(fixture.text, language: fixture.language, cangjie: cangjie)
            #expect(prepared == fixture.prepared, "\(fixture.language): \(fixture.text)")
        }
    }

    @Test func preparedTextMatchesPythonWithoutCangjie() throws {
        for fixture in try loadV3Fixtures() where fixture.language != "zh" {
            let prepared = prepareV3Text(fixture.text, language: fixture.language, cangjie: nil)
            #expect(prepared == fixture.prepared, "\(fixture.language): \(fixture.text)")
        }
    }

    @Test(.enabled(if: v3ModelDir != nil))
    func tokenIdsMatchPython() async throws {
        let modelDir = try #require(v3ModelDir)

        // swift-transformers needs tokenizer_config.json; the model writes a minimal one too.
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("chatterbox-v3-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.copyItem(
            at: modelDir.appendingPathComponent("tokenizer.json"),
            to: folder.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: ["tokenizer_class": "GPT2Tokenizer"])
            .write(to: folder.appendingPathComponent("tokenizer_config.json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)

        for fixture in try loadV3Fixtures() {
            #expect(
                chatterboxEncode(fixture.prepared, tokenizer: tokenizer) == fixture.ids,
                "\(fixture.language): \(fixture.text)")
        }
    }

    @Test func configDecodingDetectsV3() throws {
        let v3 = """
            {
              "model_type": "chatterbox", "version": "1.0", "t3_model": "v3",
              "multilingual": true, "vocab_size": 2454, "text_preprocessing": "NFKD,fullcase",
              "t3_config": {
                "text_tokens_dict_size": 2454, "start_text_token": 255, "stop_text_token": 0,
                "max_text_tokens": 2048, "speech_tokens_dict_size": 8194,
                "start_speech_token": 6561, "stop_speech_token": 6562, "max_speech_tokens": 4096,
                "llama_config_name": "Llama_520M", "input_pos_emb": "learned",
                "speech_cond_prompt_len": 150, "encoder_type": "voice_encoder",
                "speaker_embed_size": 256, "use_perceiver_resampler": true, "emotion_adv": true
              }
            }
            """
        let config = try JSONDecoder().decode(ChatterboxConfiguration.self, from: Data(v3.utf8))
        #expect(config.t3Model == "v3")
        #expect(config.multilingual == true)
        #expect(config.textPreprocessing == "NFKD,fullcase")
        #expect(config.usesV3TextPreprocessing)

        let legacy = try JSONDecoder().decode(
            ChatterboxConfiguration.self, from: Data(#"{"model_type": "chatterbox"}"#.utf8))
        #expect(legacy.t3Model == nil)
        #expect(!legacy.usesV3TextPreprocessing)

        let v3NoPreprocessing = try JSONDecoder().decode(
            ChatterboxConfiguration.self, from: Data(#"{"t3_model": "v3"}"#.utf8))
        #expect(v3NoPreprocessing.usesV3TextPreprocessing)

        let v2 = try JSONDecoder().decode(
            ChatterboxConfiguration.self,
            from: Data(#"{"t3_model": "v2", "text_preprocessing": "legacy"}"#.utf8))
        #expect(!v2.usesV3TextPreprocessing)
    }
}
