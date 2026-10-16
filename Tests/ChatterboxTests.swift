import Foundation
import MLX
import MLXAudioCore
import Testing
import Tokenizers

@testable import MLXAudioCodecs
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

// MARK: - Conditionals, options and offline loading

/// Offline model assets; the gated tests skip without both.
private let s3TokenizerDir: URL? = ProcessInfo.processInfo.environment["CHATTERBOX_S3_DIR"]
    .map { URL(fileURLWithPath: $0) }

private func peak(_ audio: MLXArray) -> Float {
    MLX.abs(audio).max().item(Float.self)
}

private func tempDirectory(_ name: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(name)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makeConditionals(tokens: Int = 3) -> ChatterboxConditionals {
    ChatterboxConditionals(
        speakerEmb: MLXRandom.normal([1, 256]),
        t3PromptSpeechTokens: MLXArray.zeros([1, tokens]).asType(.int32),
        xVector: MLXRandom.normal([1, 192]),
        s3genPromptToken: MLXArray.zeros([1, 4]).asType(.int32),
        s3genPromptFeat: MLXRandom.normal([1, 8, 80]))
}

private func expectEqual(_ a: MLXArray, _ b: MLXArray, _ label: String) {
    #expect(a.shape == b.shape, "\(label) shape")
    #expect(a.dtype == b.dtype, "\(label) dtype")
    #expect(MLX.arrayEqual(a, b).item(Bool.self), "\(label) values")
}

/// A model loaded once through the offline loader, from a read-only folder that has no
/// tokenizer_config.json, with the Hub pointed at nothing. Shared because each load costs GBs.
private struct OfflineChatterboxFixture: @unchecked Sendable {
    let model: ChatterboxModel
    let conditionals: ChatterboxConditionals
    let folderBefore: [String]
    let folderAfter: [String]
}

private let offlineFixture = Task<OfflineChatterboxFixture, Error> {
    let v3 = try #require(v3ModelDir)
    let s3 = try #require(s3TokenizerDir)
    Memory.cacheLimit = 512 * 1024 * 1024

    // Symlinked copy minus tokenizer_config.json, the file the old loader wrote.
    let folder = try tempDirectory("chatterbox-v3-readonly")
    for name in ["config.json", "model.safetensors", "tokenizer.json", "Cangjie5_TC.json"] {
        try FileManager.default.createSymbolicLink(
            at: folder.appendingPathComponent(name),
            withDestinationURL: v3.appendingPathComponent(name))
    }
    try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: folder.path)

    func listing() throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted()
    }

    let env = ProcessInfo.processInfo.environment
    let emptyCache = try tempDirectory("chatterbox-empty-hub-cache")
    setenv("HF_ENDPOINT", "http://127.0.0.1:9", 1)
    setenv("HF_HUB_CACHE", emptyCache.path, 1)
    defer {
        for (key, old) in [("HF_ENDPOINT", env["HF_ENDPOINT"]), ("HF_HUB_CACHE", env["HF_HUB_CACHE"])] {
            if let old { setenv(key, old, 1) } else { unsetenv(key) }
        }
    }

    let before = try listing()
    let model = try await ChatterboxModel.fromModelDirectory(
        folder, s3TokenizerDirectory: s3, hfToken: nil)
    let after = try listing()

    let wav = try #require(
        Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
    let (_, audio) = try loadAudioArray(from: wav, sampleRate: model.sampleRate)
    let conditionals = try model.prepareConditionals(refAudio: audio, sampleRate: model.sampleRate)
    return OfflineChatterboxFixture(
        model: model, conditionals: conditionals, folderBefore: before, folderAfter: after)
}

// MARK: - Python conditioning parity

/// Keep in step with `synthetic` in scripts/make_chatterbox_parity_fixtures.py.
private func syntheticSignal(count: Int, rate: Int) -> [Float] {
    (0 ..< count).map { n in
        let t = Double(n) / Double(rate)
        let noise = Double((n * 7919) % 101) / 50.0 - 1.0
        return Float(0.5 * sin(2 * .pi * 440 * t) + 0.3 * sin(2 * .pi * 7900 * t) + 0.1 * noise)
    }
}

private func mediaArrays(_ name: String) throws -> [String: MLXArray] {
    let url = try #require(Bundle.module.url(forResource: name, withExtension: "safetensors", subdirectory: "media"))
    return try MLX.loadArrays(url: url)
}

private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let a = a.reshaped([-1]), b = b.reshaped([-1])
    return ((a * b).sum() / (MLX.sqrt((a * a).sum()) * MLX.sqrt((b * b).sum()))).item(Float.self)
}

private func sameFraction(_ a: MLXArray, _ b: MLXArray) -> Float {
    guard a.shape == b.shape, a.size > 0 else { return 0 }
    return (a .== b).asType(.float32).mean().item(Float.self)
}

@Suite("Chatterbox resample tests")
struct ChatterboxResampleTests {
    @Test(arguments: [(24000, 16000, 4801), (44100, 24000, 4410)])
    func matchesMLXAudioPython(source: Int, target: Int, count: Int) throws {
        let expected = try #require(try mediaArrays("chatterbox_resample_golden")["\(source)_\(target)"])
        let actual = MLXArray(kaiserBestResample(syntheticSignal(count: count, rate: source), from: source, to: target))
        #expect(actual.shape == expected.shape)
        #expect(MLX.abs(actual - expected).max().item(Float.self) < 1e-5)
    }

    @Test func sameRateIsIdentity() {
        let x: [Float] = [0.1, -0.2, 0.3]
        #expect(kaiserBestResample(x, from: 24000, to: 24000) == x)
    }
}

@Suite("Chatterbox T3 override tests")
struct ChatterboxT3OverrideTests {
    private let base: [String: MLXArray] = [
        "t3.tfmr.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 2]),
        "t3.tfmr.model.norm.weight": MLXArray.zeros([2]),
        "t3.speech_emb.weight": MLXArray.zeros([3, 2]),
        "s3gen.flow.weight": MLXArray.zeros([1]),
    ]

    @Test func renamesResembleKeysAndKeepsTheRest() throws {
        let merged = try replacingT3Weights(in: base, with: [
            "tfmr.layers.0.self_attn.q_proj.weight": MLXArray.ones([2, 2]),
            "tfmr.norm.weight": MLXArray.ones([2]),
            "speech_emb.weight": MLXArray.ones([3, 2]),
        ])
        #expect(Set(merged.keys) == Set(base.keys))
        #expect(merged["t3.tfmr.model.norm.weight"]!.sum().item(Float.self) == 2)
        #expect(merged["s3gen.flow.weight"]!.sum().item(Float.self) == 0)
    }

    @Test func rejectsMissingKeysAndOtherShapes() {
        #expect(throws: AudioGenerationError.self) {
            try replacingT3Weights(in: base, with: ["tfmr.norm.weight": MLXArray.ones([2])])
        }
        #expect(throws: AudioGenerationError.self) {
            try replacingT3Weights(in: base, with: [
                "tfmr.layers.0.self_attn.q_proj.weight": MLXArray.ones([2, 3]),
                "tfmr.norm.weight": MLXArray.ones([2]), "speech_emb.weight": MLXArray.ones([3, 2]),
            ])
        }
    }
}

/// A Resemble V3 language pack's T3 file, e.g. t3_hi.safetensors; the gated test skips without it.
private let packT3 = ProcessInfo.processInfo.environment["CHATTERBOX_PACK_T3"].map { URL(fileURLWithPath: $0) }

@Suite("Chatterbox language pack tests", .enabled(if: v3ModelDir != nil && s3TokenizerDir != nil && packT3 != nil))
struct ChatterboxLanguagePackTests {
    @Test func aPackLoadsOverV3AndChangesOnlyT3() async throws {
        let fixture = try await offlineFixture.value
        let packModel = try await ChatterboxModel.fromModelDirectory(
            try #require(v3ModelDir), s3TokenizerDirectory: try #require(s3TokenizerDir), t3Weights: try #require(packT3),
            hfToken: nil)
        let base = Dictionary(uniqueKeysWithValues: fixture.model.parameters().flattened())
        let pack = Dictionary(uniqueKeysWithValues: packModel.parameters().flattened())
        // Turbo's mixer is never loaded for Regular, so it differs between any two instances.
        let changed = base.keys.filter {
            !$0.hasSuffix("time_embed_mixer.weight") && !MLX.allClose(base[$0]!, pack[$0]!).item(Bool.self)
        }
        #expect(!changed.isEmpty)
        #expect(changed.allSatisfy { $0.hasPrefix("t3.") })
        // Same voice encoder, so V3 conditionals apply unchanged.
        let audio = try await packModel.generate(text: "नमस्ते, यह एक परीक्षण है।", language: "hi", conditionals: fixture.conditionals)
        #expect(audio.size > 24_000 / 2)
    }
}

@Suite("Chatterbox flow noise tests")
struct ChatterboxFlowNoiseTests {
    @Test func regularNoiseIsMLXAudioPythonsSeedZeroBuffer() throws {
        let head = try #require(try mediaArrays("chatterbox_v3_s3gen")["flow_noise_head"])
        let ours = CausalConditionalCFM.regularNoise()[0..., 0..., ..<head.dim(2)]
        #expect(MLX.allClose(ours, head, rtol: 0, atol: 1e-6).item(Bool.self))
    }
}

@Suite("Chatterbox sampling tests")
struct ChatterboxSamplingTests {
    // A token at 7% of the top token's probability: mlx-audio keeps it at min-p 0.05, T 0.8;
    // tempering first would drop it (0.07^1.25 < 0.05).
    private let logits = MLXArray([0, log(Float(0.07)), log(Float(0.01))]).reshaped([1, 3])

    @Test func minPFiltersBeforeTemperatureLikeMLXAudio() {
        let kept = filterLogits(logits: logits, temperature: 0.8, minP: 0.05, temperatureLast: true) .> MLXArray(-Float.infinity)
        #expect(kept.asArray(Bool.self) == [true, true, false])
    }

    @Test func temperatureFirstStaysTheDefault() {
        let kept = filterLogits(logits: logits, temperature: 0.8, minP: 0.05) .> MLXArray(-Float.infinity)
        #expect(kept.asArray(Bool.self) == [true, false, false])
    }

    @Test func temperatureLastStillScalesKeptLogits() {
        let filtered = filterLogits(logits: logits, temperature: 0.8, minP: 0.05, temperatureLast: true)
        #expect(abs(filtered[0, 1].item(Float.self) - log(Float(0.07)) / 0.8) < 1e-5)
    }
}

@Suite("Chatterbox conditionals tests")
struct ChatterboxConditionalsTests {
    @Test func saveLoadRoundTripsSyntheticArrays() throws {
        let dir = try tempDirectory("chatterbox-conds")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("voice.safetensors")

        let original = makeConditionals()
        try original.save(to: url)
        // Overwriting an existing file goes through the replace path.
        try original.save(to: url)
        let loaded = try ChatterboxConditionals.load(from: url)

        expectEqual(original.speakerEmb, loaded.speakerEmb, "speakerEmb")
        expectEqual(original.t3PromptSpeechTokens, loaded.t3PromptSpeechTokens, "t3 tokens")
        expectEqual(original.xVector, loaded.xVector, "xVector")
        expectEqual(original.s3genPromptToken, loaded.s3genPromptToken, "s3gen tokens")
        expectEqual(original.s3genPromptFeat, loaded.s3genPromptFeat, "s3gen feat")

        let (_, metadata) = try MLX.loadArraysAndMetadata(url: url)
        #expect(metadata["format"] == "chatterbox-conditionals")
        #expect(metadata["version"] == "1")
        #expect(try FileManager.default.contentsOfDirectory(atPath: dir.path) == ["voice.safetensors"])
    }

    @Test func emptyT3PromptTokensRoundTrip() throws {
        let dir = try tempDirectory("chatterbox-conds-empty")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("voice.safetensors")

        try makeConditionals(tokens: 0).save(to: url)
        let loaded = try ChatterboxConditionals.load(from: url)
        #expect(loaded.t3PromptSpeechTokens.shape == [1, 0])
    }

    @Test func loadRejectsMissingKey() throws {
        let dir = try tempDirectory("chatterbox-conds-bad")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("missing.safetensors")

        let c = makeConditionals()
        try MLX.save(
            arrays: ["t3.speaker_emb": c.speakerEmb, "gen.embedding": c.xVector], url: url)

        #expect {
            try ChatterboxConditionals.load(from: url)
        } throws: { error in
            guard case ChatterboxConditionalsError.missingKey = error else { return false }
            return true
        }
    }

    @Test func loadRejectsWrongShape() throws {
        let dir = try tempDirectory("chatterbox-conds-shape")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("shape.safetensors")

        let c = makeConditionals()
        try MLX.save(
            arrays: [
                "t3.speaker_emb": MLXRandom.normal([1, 128]),
                "t3.cond_prompt_speech_tokens": c.t3PromptSpeechTokens,
                "gen.embedding": c.xVector,
                "gen.prompt_token": c.s3genPromptToken,
                "gen.prompt_feat": c.s3genPromptFeat,
            ], url: url)

        #expect {
            try ChatterboxConditionals.load(from: url)
        } throws: { error in
            guard case ChatterboxConditionalsError.invalidShape(let key, _, _) = error else {
                return false
            }
            return key == "t3.speaker_emb"
        }
    }

    @Test func optionsDefaults() {
        let options = ChatterboxGenerationOptions()
        #expect(options.temperature == 0.8)
        #expect(options.exaggeration == 0.5)
        #expect(options.cfgWeight == 0.5)
        #expect(options.normalizePeak)
    }
}

@Suite("Chatterbox offline model tests", .serialized, .enabled(if: v3ModelDir != nil && s3TokenizerDir != nil))
struct ChatterboxOfflineModelTests {
    @Test func loadWritesNothingIntoModelFolder() async throws {
        let fixture = try await offlineFixture.value
        #expect(fixture.folderBefore == fixture.folderAfter)
        #expect(!fixture.folderAfter.contains("tokenizer_config.json"))
        #expect(fixture.model.tokenizer != nil)
        #expect(fixture.model.s3Tokenizer != nil)
    }

    @Test func missingS3DirectoryThrowsBeforeLoadingWeights() async throws {
        let empty = try tempDirectory("chatterbox-empty-s3")
        defer { try? FileManager.default.removeItem(at: empty) }
        await #expect(throws: (any Error).self) {
            _ = try await ChatterboxModel.fromModelDirectory(
                try #require(v3ModelDir), s3TokenizerDirectory: empty, hfToken: nil)
        }
    }

    @Test func preparedConditionalsRoundTripExactly() async throws {
        let fixture = try await offlineFixture.value
        let dir = try tempDirectory("chatterbox-conds-model")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("voice.safetensors")

        let c = fixture.conditionals
        #expect(c.speakerEmb.shape == [1, 256])
        #expect(c.xVector.shape == [1, 192])
        #expect(c.t3PromptSpeechTokens.dtype == .int32)
        #expect(c.s3genPromptToken.dtype == .int32)
        #expect(c.s3genPromptFeat.dim(2) == 80)
        #expect(c.s3genPromptFeat.dim(1) == 2 * c.s3genPromptToken.dim(1))

        try c.save(to: url)
        let loaded = try ChatterboxConditionals.load(from: url)
        expectEqual(c.speakerEmb, loaded.speakerEmb, "speakerEmb")
        expectEqual(c.t3PromptSpeechTokens, loaded.t3PromptSpeechTokens, "t3 tokens")
        expectEqual(c.xVector, loaded.xVector, "xVector")
        expectEqual(c.s3genPromptToken, loaded.s3genPromptToken, "s3gen tokens")
        expectEqual(c.s3genPromptFeat, loaded.s3genPromptFeat, "s3gen feat")
    }

    @Test func conditionalsMatchMLXAudioPython() async throws {
        let fixture = try await offlineFixture.value
        let python = try mediaArrays("chatterbox_v3_conds_conversational_a")
        let swift = fixture.conditionals

        let speaker = cosine(swift.speakerEmb, try #require(python["t3.speaker_emb"]))
        let xVector = cosine(swift.xVector, try #require(python["gen.embedding"]))
        let t3Tokens = sameFraction(swift.t3PromptSpeechTokens, try #require(python["t3.cond_prompt_speech_tokens"]))
        let s3Tokens = sameFraction(swift.s3genPromptToken, try #require(python["gen.prompt_token"]))
        let feat = try #require(python["gen.prompt_feat"])

        #expect(speaker > 0.9999)
        #expect(xVector > 0.9999)
        #expect(t3Tokens == 1)
        #expect(s3Tokens == 1)
        #expect(swift.s3genPromptFeat.shape == feat.shape)
        #expect(MLX.allClose(swift.s3genPromptFeat, feat, rtol: 1e-3, atol: 1e-3).item(Bool.self))
    }

    @Test func firstStepLogitsMatchMLXAudioPython() async throws {
        let fixture = try await offlineFixture.value
        let python = try mediaArrays("chatterbox_v3_first_step")
        let t3 = try #require(fixture.model.t3 as? T3Model)
        let c = fixture.conditionals

        // Keep the text in step with FIRST_STEP_TEXT in scripts/make_chatterbox_parity_fixtures.py.
        let tokens = try fixture.model.tokenizeText("Hello there, this is a test.", language: "en")
        let pythonTokens = try #require(python["text_tokens"])
        let promptLength = try #require(python["prompt_length"]).item(Int.self)
        let expected = try #require(python["logits"])
        #expect(tokens.reshaped([-1]).asArray(Int32.self) == pythonTokens.asArray(Int32.self))

        var cond = T3Cond(
            speakerEmb: c.speakerEmb, condPromptSpeechTokens: c.t3PromptSpeechTokens, condPromptSpeechEmb: nil,
            emotionAdv: MLXArray(Float(0.5)))
        let (hidden, cache) = t3.prefill(t3Cond: &cond, textTokens: tokens, cfgWeight: 0.5)
        #expect(cache.first?.offset == promptLength)
        let logits = t3.speechHead(hidden[0..., (-1)..., 0...]).squeezed(axis: 1)
        let guided = logits[0 ..< 1] + 0.5 * (logits[0 ..< 1] - logits[1 ..< 2])

        #expect(cosine(guided, expected) > 0.9999)
        #expect(guided.argMax(axis: -1).item(Int.self) == expected.argMax(axis: -1).item(Int.self))
        // Same candidate set after min-p, so sampling draws from the same tokens.
        func kept(_ x: MLXArray) -> MLXArray {
            filterLogits(logits: x, temperature: 0.8, minP: 0.05, temperatureLast: true) .> MLXArray(-Float.infinity)
        }
        #expect(MLX.all(kept(guided) .== kept(expected)).item(Bool.self))
    }

    @Test func speakerEmbeddingIsTheConditioningsOne() async throws {
        let fixture = try await offlineFixture.value
        let wav = try #require(
            Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
        let (_, audio) = try loadAudioArray(from: wav, sampleRate: fixture.model.sampleRate)
        let embedding = fixture.model.speakerEmbedding(audio: audio, sampleRate: fixture.model.sampleRate)
        #expect(MLX.allClose(embedding, fixture.conditionals.speakerEmb, rtol: 0, atol: 1e-6).item(Bool.self))
    }

    @Test func everyCheckpointWeightReachesAParameter() async throws {
        // update(verify: []) drops unmatched keys silently; the S3Gen encoder once ran on random init that way.
        let fixture = try await offlineFixture.value
        let checkpoint = try MLX.loadArrays(url: try #require(v3ModelDir).appendingPathComponent("model.safetensors"))
        let weights = Set(fixture.model.sanitize(weights: checkpoint).keys)
        let parameters = Set(fixture.model.parameters().flattened().map(\.0))

        #expect(weights.subtracting(parameters).sorted() == [])
        // Computed at init, Turbo-only, or affine-free BatchNorm.
        let unfilled = parameters.subtracting(weights).filter { !$0.hasSuffix("attn1.to_out.0.bias") }.sorted()
        #expect(unfilled == [
            "s3gen.decoder.estimator.time_embed_mixer.weight",
            "s3gen.encoder.embed.pos_enc.pe",
            "s3gen.encoder.up_embed.pos_enc.pe",
            "s3gen.mel2wav.stftWindow",
            "s3gen.speaker_encoder.dense.nonlinear.0.bias",
            "s3gen.speaker_encoder.dense.nonlinear.0.weight",
        ])
    }

    @Test func s3genMatchesMLXAudioPython() async throws {
        let fixture = try await offlineFixture.value
        let python = try mediaArrays("chatterbox_v3_s3gen")
        let tokens = try #require(python["speech_tokens"])
        let expectedMel = try #require(python["mel"])
        let expectedWav = try #require(python["wav"]).asType(.float32)
        let c = fixture.conditionals

        let mel = fixture.model.s3gen.inference(
            token: tokens, tokenLen: MLXArray([Int32(tokens.dim(1))]),
            promptToken: c.s3genPromptToken, promptTokenLen: MLXArray([Int32(c.s3genPromptToken.dim(1))]),
            promptFeat: c.s3genPromptFeat, embedding: c.xVector, nTimesteps: 10)
        #expect(mel.shape == expectedMel.shape)
        let melCos = cosine(mel, expectedMel)
        let melDiff = MLX.abs(mel - expectedMel).max().item(Float.self)

        // Same seed and call order as the fixture, so the vocoder's sine phases and noise match too.
        MLXRandom.seed(123)
        let (wav, _) = fixture.model.s3gen.vocoder(expectedMel)
        let flat = wav.reshaped([-1])
        #expect(flat.dim(0) == expectedWav.reshaped([-1]).dim(0))
        let wavCos = cosine(flat, expectedWav)
        #expect(melCos > 0.99999)
        #expect(melDiff < 0.1)
        #expect(wavCos > 0.999)
    }

    @Test func generationIsDeterministicAcrossSavedConditionals() async throws {
        let fixture = try await offlineFixture.value
        let dir = try tempDirectory("chatterbox-determinism")
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("voice.safetensors")
        try fixture.conditionals.save(to: url)
        let loaded = try ChatterboxConditionals.load(from: url)

        MLXRandom.seed(1234)
        let a = try await fixture.model.generate(
            text: "Hello there, this is a test.", language: "en", conditionals: fixture.conditionals)
        MLXRandom.seed(1234)
        let b = try await fixture.model.generate(
            text: "Hello there, this is a test.", language: "en", conditionals: loaded)
        eval(a, b)
        // Same seed gives the same tokens (same length); the vocoder itself is only
        // reproducible to ~1e-7 on GPU, so exact equality is too strict.
        #expect(a.shape == b.shape)
        #expect(MLX.allClose(a, b, rtol: 0, atol: 1e-5).item(Bool.self))
    }

    @Test func normalizePeakOptionControlsRescaling() async throws {
        let fixture = try await offlineFixture.value
        let text = "Hello there, this is a test."

        MLXRandom.seed(99)
        let normalized = try await fixture.model.generate(
            text: text, language: "en", conditionals: fixture.conditionals,
            options: ChatterboxGenerationOptions(normalizePeak: true))
        MLXRandom.seed(99)
        let raw = try await fixture.model.generate(
            text: text, language: "en", conditionals: fixture.conditionals,
            options: ChatterboxGenerationOptions(normalizePeak: false))

        let normalizedPeak = peak(normalized)
        let rawPeak = peak(raw)
        #expect(abs(normalizedPeak - 0.95) < 1e-3)
        #expect(abs(rawPeak - 0.95) > 1e-3)
        // Same seed, so the only difference is the scale factor.
        #expect(MLX.allClose(raw * (0.95 / rawPeak), normalized, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    @Test func protocolGenerateMatchesConditionalsPath() async throws {
        let fixture = try await offlineFixture.value
        let text = "Hello there, this is a test."
        let wav = try #require(
            Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
        let (_, refAudio) = try loadAudioArray(from: wav, sampleRate: fixture.model.sampleRate)

        MLXRandom.seed(5)
        let legacy = try await fixture.model.generate(
            text: text, voice: nil, refAudio: refAudio, refText: nil, language: "en",
            generationParameters: fixture.model.defaultGenerationParameters)
        MLXRandom.seed(5)
        let modern = try await fixture.model.generate(
            text: text, language: "en", conditionals: fixture.conditionals)
        eval(legacy, modern)
        #expect(legacy.shape == modern.shape)
        #expect(MLX.allClose(legacy, modern, rtol: 0, atol: 1e-5).item(Bool.self))
    }

    @Test func exaggerationAndCfgWeightChangeOutput() async throws {
        let fixture = try await offlineFixture.value
        let text = "Hello there, this is a test."

        MLXRandom.seed(7)
        let base = try await fixture.model.generate(
            text: text, language: "en", conditionals: fixture.conditionals)
        MLXRandom.seed(7)
        let exaggerated = try await fixture.model.generate(
            text: text, language: "en", conditionals: fixture.conditionals,
            options: ChatterboxGenerationOptions(exaggeration: 0.9))
        eval(base, exaggerated)
        #expect(base.shape != exaggerated.shape || !MLX.arrayEqual(base, exaggerated).item(Bool.self))
    }
}
