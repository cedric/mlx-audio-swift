//
//  ChatterboxConditionals.swift
//  MLXAudio
//
//  Storable per-reference conditioning for Chatterbox, plus per-call generation options.
//

import Foundation
@preconcurrency import MLX

// MARK: - Errors

public enum ChatterboxConditionalsError: LocalizedError {
    case missingKey(String)
    case invalidShape(key: String, expected: String, actual: [Int])
    case unsupportedFormat(String)

    public var errorDescription: String? {
        switch self {
        case .missingKey(let key):
            return "Chatterbox conditionals file is missing '\(key)'"
        case .invalidShape(let key, let expected, let actual):
            return "Chatterbox conditionals '\(key)' has shape \(actual), expected \(expected)"
        case .unsupportedFormat(let detail):
            return "Unsupported Chatterbox conditionals file: \(detail)"
        }
    }
}

// MARK: - Conditionals

/// Everything generation needs from a reference clip, computed once by
/// `ChatterboxModel.prepareConditionals(refAudio:sampleRate:)` and reusable across calls.
///
/// Emotion exaggeration is deliberately absent: it is applied per generation.
public struct ChatterboxConditionals: @unchecked Sendable {
    // Immutable arrays are only read after creation, hence the unchecked conformance.

    /// T3 speaker embedding, (1, 256).
    public let speakerEmb: MLXArray
    /// T3 prompt speech tokens, (1, T) int32; T may be 0.
    public let t3PromptSpeechTokens: MLXArray
    /// S3Gen x-vector, (1, 192).
    public let xVector: MLXArray
    /// S3Gen prompt speech tokens, (1, N) int32; N may be 0.
    public let s3genPromptToken: MLXArray
    /// S3Gen prompt mel features in the layout the flow encoder consumes, (1, T', 80) with T' == 2N.
    public let s3genPromptFeat: MLXArray

    public init(
        speakerEmb: MLXArray,
        t3PromptSpeechTokens: MLXArray,
        xVector: MLXArray,
        s3genPromptToken: MLXArray,
        s3genPromptFeat: MLXArray
    ) {
        self.speakerEmb = speakerEmb
        self.t3PromptSpeechTokens = t3PromptSpeechTokens
        self.xVector = xVector
        self.s3genPromptToken = s3genPromptToken
        self.s3genPromptFeat = s3genPromptFeat
    }

    // Same keys as the `conds.safetensors` shipped with model repos.
    private enum Key {
        static let speakerEmb = "t3.speaker_emb"
        static let t3PromptSpeechTokens = "t3.cond_prompt_speech_tokens"
        static let xVector = "gen.embedding"
        static let s3genPromptToken = "gen.prompt_token"
        static let s3genPromptFeat = "gen.prompt_feat"
    }

    private static let formatName = "chatterbox-conditionals"
    private static let formatVersion = 1

    /// Writes a safetensors file atomically (temp file in the same directory, then rename).
    public func save(to url: URL) throws {
        let arrays: [String: MLXArray] = [
            Key.speakerEmb: speakerEmb,
            Key.t3PromptSpeechTokens: t3PromptSpeechTokens,
            Key.xVector: xVector,
            Key.s3genPromptToken: s3genPromptToken,
            Key.s3genPromptFeat: s3genPromptFeat,
        ]
        let fm = FileManager.default
        // MLX appends ".safetensors" to any other extension, so the temp name must end with it.
        let tmp = url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp.safetensors")
        do {
            try MLX.save(
                arrays: arrays,
                metadata: ["format": Self.formatName, "version": String(Self.formatVersion)],
                url: tmp)
            if fm.fileExists(atPath: url.path) {
                _ = try fm.replaceItemAt(url, withItemAt: tmp)
            } else {
                try fm.moveItem(at: tmp, to: url)
            }
        } catch {
            try? fm.removeItem(at: tmp)
            throw error
        }
    }

    /// Reads a file written by `save(to:)`, rejecting missing keys and wrong shapes.
    public static func load(from url: URL) throws -> ChatterboxConditionals {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)

        // Metadata is optional so a model repo's own conds.safetensors also loads.
        if let format = metadata["format"], format != formatName {
            throw ChatterboxConditionalsError.unsupportedFormat("format '\(format)'")
        }
        if let version = metadata["version"], (Int(version) ?? Int.max) > formatVersion {
            throw ChatterboxConditionalsError.unsupportedFormat("version '\(version)'")
        }

        func array(_ key: String, ndim: Int, expected: String, check: ([Int]) -> Bool) throws
            -> MLXArray
        {
            guard let value = arrays[key] else { throw ChatterboxConditionalsError.missingKey(key) }
            guard value.ndim == ndim, check(value.shape) else {
                throw ChatterboxConditionalsError.invalidShape(
                    key: key, expected: expected, actual: value.shape)
            }
            return value
        }

        let speakerEmb = try array(Key.speakerEmb, ndim: 2, expected: "[1, 256]") {
            $0 == [1, 256]
        }
        let t3Tokens = try array(Key.t3PromptSpeechTokens, ndim: 2, expected: "[1, T]") {
            $0[0] == 1
        }
        let xVector = try array(Key.xVector, ndim: 2, expected: "[1, 192]") { $0 == [1, 192] }
        let promptToken = try array(Key.s3genPromptToken, ndim: 2, expected: "[1, N]") {
            $0[0] == 1
        }
        let promptFeat = try array(Key.s3genPromptFeat, ndim: 3, expected: "[1, T', 80]") {
            $0[0] == 1 && $0[2] == 80
        }

        let result = ChatterboxConditionals(
            speakerEmb: speakerEmb,
            t3PromptSpeechTokens: t3Tokens.asType(.int32),
            xVector: xVector,
            s3genPromptToken: promptToken.asType(.int32),
            s3genPromptFeat: promptFeat)
        // Safetensors loads are lazy; materialize before the file can go away.
        eval(
            result.speakerEmb, result.t3PromptSpeechTokens, result.xVector,
            result.s3genPromptToken, result.s3genPromptFeat)
        return result
    }
}

// MARK: - Options

/// Per-call generation settings for `ChatterboxModel.generate(text:language:conditionals:options:)`.
public struct ChatterboxGenerationOptions: Sendable {
    public var temperature: Float
    /// Emotion exaggeration fed to T3 as `emotion_adv`.
    public var exaggeration: Float
    /// Classifier-free guidance weight (Regular model only).
    public var cfgWeight: Float
    /// Scale the output to a 0.95 peak. Disable when stitching chunks.
    public var normalizePeak: Bool

    public init(
        temperature: Float = 0.8,
        exaggeration: Float = 0.5,
        cfgWeight: Float = 0.5,
        normalizePeak: Bool = true
    ) {
        self.temperature = temperature
        self.exaggeration = exaggeration
        self.cfgWeight = cfgWeight
        self.normalizePeak = normalizePeak
    }
}
