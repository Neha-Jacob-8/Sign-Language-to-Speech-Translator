// Sign classifier that runs in the browser.
// Same features and network as signspeak/features.py and signspeak/classifier.py,
// loaded from the files written by training/train.py (models/sign_mlp.json + .bin).

const NUM_LANDMARKS = 21;
const MAX_HANDS = 2;
const WRIST = 0;
const MIDDLE_MCP = 9;
export const FEATURE_VERSION = 1;
export const FEATURE_DIM = MAX_HANDS * NUM_LANDMARKS * 3 + MAX_HANDS + 3;

// hands: up to two hands of 21 [x, y, z] points in MediaPipe's normalised image coordinates.
// See signspeak/features.py for the layout of the 131 values.
export function featurize(hands, width, height) {
  const px = hands.map((hand) => hand.map(([x, y, z]) => [x * width, y * height, z * width]));
  px.sort((a, b) => a[WRIST][0] - b[WRIST][0]); // left-most hand first
  const slots = px.slice(0, MAX_HANDS);
  const f = new Float32Array(FEATURE_DIM);
  const palms = [];
  slots.forEach((hand, s) => {
    const wrist = hand[WRIST];
    const mcp = hand[MIDDLE_MCP];
    const palm = Math.max(Math.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1], mcp[2] - wrist[2]), 1e-6);
    palms.push(palm);
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      for (let k = 0; k < 3; k++) f[s * 63 + i * 3 + k] = (hand[i][k] - wrist[k]) / palm;
    }
    f[126 + s] = 1;
  });
  if (slots.length === 2) {
    const meanPalm = (palms[0] + palms[1]) / 2;
    for (let k = 0; k < 3; k++) f[128 + k] = (slots[1][WRIST][k] - slots[0][WRIST][k]) / meanPalm;
  }
  for (let i = 0; i < FEATURE_DIM; i++) f[i] = Math.min(20, Math.max(-20, f[i]));
  return f;
}

// Abramowitz & Stegun 7.1.26 (max error 1.5e-7), as in signspeak/classifier.py.
function erf(x) {
  const sign = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-a * a));
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class SignModel {
  constructor(meta, buffer) {
    if (meta.feature_version !== FEATURE_VERSION) throw new Error(`Model feature version ${meta.feature_version} != ${FEATURE_VERSION}`);
    const weights = new Float32Array(buffer);
    this.classes = meta.classes;
    this.metrics = meta.metrics || {};
    this.layers = meta.layers.map((l) => ({
      nIn: l.in,
      nOut: l.out,
      w: weights.subarray(l.weight_offset, l.weight_offset + l.in * l.out),
      b: weights.subarray(l.bias_offset, l.bias_offset + l.out),
    }));
  }

  static async load(url) {
    const jsonUrl = new URL(url, document.baseURI);
    const meta = await (await fetch(jsonUrl)).json();
    const buffer = await (await fetch(new URL(meta.weights, jsonUrl))).arrayBuffer();
    return new SignModel(meta, buffer);
  }

  static fromBuffers(meta, buffer) {
    return new SignModel(meta, buffer);
  }

  probabilities(features) {
    let x = features;
    this.layers.forEach(({ nIn, nOut, w, b }, li) => {
      const y = new Float32Array(nOut);
      const last = li === this.layers.length - 1;
      for (let o = 0; o < nOut; o++) {
        let s = b[o];
        const row = o * nIn;
        for (let i = 0; i < nIn; i++) s += w[row + i] * x[i];
        y[o] = last ? s : gelu(s);
      }
      x = y;
    });
    const max = Math.max(...x);
    const exp = x.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, v) => a + v, 0);
    return exp.map((v) => v / sum);
  }

  // Same shape as the API response: {label, confidence, top: [{label, prob}], num_hands}
  predict(hands, width, height, topK = 3) {
    if (!hands.length) return { label: null, confidence: 0, top: [], num_hands: 0 };
    const probs = this.probabilities(featurize(hands, width, height));
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, topK);
    const top = order.map((i) => ({ label: this.classes[i], prob: Math.round(probs[i] * 1e4) / 1e4 }));
    return { label: top[0].label, confidence: probs[order[0]], top, num_hands: Math.min(hands.length, MAX_HANDS) };
  }
}
