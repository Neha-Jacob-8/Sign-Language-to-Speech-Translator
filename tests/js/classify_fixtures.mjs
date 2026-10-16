// Runs frontend/classifier.js on tests/fixtures/landmarks.json and prints features + probabilities as JSON.
// Used by tests/test_web_classifier.py to check that the browser and Python agree.
import { readFileSync } from "node:fs";
import { SignModel, featurize } from "../../frontend/classifier.js";

const root = new URL("../../", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("models/sign_mlp.json", root), "utf8"));
const bin = readFileSync(new URL(`models/${meta.weights}`, root));
const model = SignModel.fromBuffers(meta, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const fixtures = JSON.parse(readFileSync(new URL("tests/fixtures/landmarks.json", root), "utf8"));

const out = fixtures.map((s) => {
  const features = featurize(s.hands, s.width, s.height);
  return {
    features: Array.from(features),
    probabilities: Array.from(model.probabilities(features)),
    label: model.predict(s.hands, s.width, s.height).label,
  };
});
process.stdout.write(JSON.stringify(out));
