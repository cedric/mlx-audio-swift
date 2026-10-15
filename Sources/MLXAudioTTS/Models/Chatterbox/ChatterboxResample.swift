import Accelerate
import Foundation

/// Port of mlx-audio's `resample_audio_array`: `scipy.signal.resample_poly` with a kaiser_best
/// FIR (64 zero crossings, rolloff 0.9476, β 14.77) and `padtype="edge"`. Chatterbox conditioning
/// uses it so voices match the Python reference.
public func kaiserBestResample(_ x: [Float], from sourceRate: Int, to targetRate: Int) -> [Float] {
    guard sourceRate != targetRate, !x.isEmpty else { return x }
    let g = gcd(sourceRate, targetRate)
    let up = targetRate / g
    let down = sourceRate / g
    let maxRate = max(up, down)

    var h = kaiserLowpass(numTaps: 2 * 64 * maxRate + 1, cutoff: 0.9475937167399596 / Double(maxRate),
                          beta: 14.769656459379492).map { $0 * Double(up) }

    // resample_poly's alignment: pad h so output samples sit at the filter centre.
    let nIn = x.count
    let nOut = (nIn * up + down - 1) / down
    let nPrePad = down - (h.count / 2) % down
    let nPreRemove = (h.count / 2 + nPrePad) / down
    var nPostPad = 0
    func outputLength(_ hLen: Int) -> Int { ((nIn - 1) * up + hLen - 1) / down + 1 }
    while outputLength(h.count + nPrePad + nPostPad) < nOut + nPreRemove { nPostPad += 1 }
    h = Array(repeating: 0, count: nPrePad) + h + Array(repeating: 0, count: nPostPad)

    // y[m] = Σ_k h[k]·xup[m·down − k], xup = edge-extended x with up−1 zeros between samples.
    // Lay out xup from m = nPreRemove so vDSP_desamp (a strided correlation) computes it with h reversed.
    let lh = h.count
    let offset = nPreRemove * down - (lh - 1)
    let bufferLen = (nOut - 1) * down + lh
    var buffer = [Double](repeating: 0, count: bufferLen)
    for t in 0 ..< bufferLen {
        let j = t + offset
        let (q, r) = j.quotientAndRemainder(dividingBy: up)
        guard r == 0 else { continue }
        buffer[t] = Double(x[min(max(q, 0), nIn - 1)])
    }
    // Double precision like scipy's upfirdn, then float32 like mlx-audio.
    let reversed = Array(h.reversed())
    var y = [Double](repeating: 0, count: nOut)
    vDSP_desampD(buffer, vDSP_Stride(down), reversed, &y, vDSP_Length(nOut), vDSP_Length(lh))
    return y.map { Float($0) }
}

/// `scipy.signal.firwin(numTaps, cutoff, window=("kaiser", beta))`: unit DC gain lowpass,
/// cutoff relative to Nyquist.
func kaiserLowpass(numTaps: Int, cutoff: Double, beta: Double) -> [Double] {
    let alpha = Double(numTaps - 1) / 2
    let i0Beta = besselI0(beta)
    var h = (0 ..< numTaps).map { n -> Double in
        let m = Double(n) - alpha
        let x = cutoff * m
        let sinc = x == 0 ? 1 : sin(Double.pi * x) / (Double.pi * x)
        let ratio = m / alpha
        let window = besselI0(beta * (max(0, 1 - ratio * ratio)).squareRoot()) / i0Beta
        return cutoff * sinc * window
    }
    let sum = h.reduce(0, +)
    for i in h.indices { h[i] /= sum }
    return h
}

private func besselI0(_ x: Double) -> Double {
    var sum = 1.0
    var term = 1.0
    let half = x / 2
    for k in 1 ... 200 {
        term *= (half / Double(k)) * (half / Double(k))
        sum += term
        if term < 1e-17 * sum { break }
    }
    return sum
}

private func gcd(_ a: Int, _ b: Int) -> Int {
    var (a, b) = (a, b)
    while b != 0 { (a, b) = (b, a % b) }
    return a
}
