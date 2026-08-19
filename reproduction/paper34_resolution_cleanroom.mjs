// Clean-room Node.js implementation of the learned factorized fit and analysis.
// This file imports no project code and uses only the Node standard library.

import fs from "node:fs";
import crypto from "node:crypto";

const [inputPath, analysisPath, ledgerPath, outputPath] = process.argv.slice(2);
if (!outputPath) {
  throw new Error("usage: node paper34_resolution_cleanroom.mjs INPUT ANALYSIS LEDGER OUTPUT");
}

const MOD = 7;
const LAYERS = 5;
const FAMILIES = ["linear_target", "quadratic_target", "source_target"];
const OOD_NOISE = 0.12;
const ALPHA = 0.05;
const EFFECT_COUNT = 8;

function combinations(values) {
  const result = [];
  for (let i = 0; i < values.length; i += 1) {
    for (let j = i + 1; j < values.length; j += 1) result.push([values[i], values[j]]);
  }
  return result;
}

const GRAPHS = [];
for (let target = 0; target < LAYERS; target += 1) {
  GRAPHS.push(...combinations([0, 1, 2, 3, 4].filter((x) => x !== target)).map((s) => [target, s]));
}

function headValue(family, source, action, sourceIndex, target) {
  const active = action[sourceIndex];
  if (family === "linear_target") return active * (1 + source[target]);
  if (family === "quadratic_target") return active * (1 + source[target]) ** 2;
  if (family === "source_target") return active * (1 + source[sourceIndex] + source[target]);
  throw new Error(`unknown family ${family}`);
}

function predict(model, source, action) {
  const [target, sources] = model.graph;
  const delta = action.map((value, index) => model.multipliers[index] * value);
  for (let edge = 0; edge < 2; edge += 1) {
    delta[target] += model.coefficients[edge] * headValue(model.families[edge], source, action, sources[edge], target);
  }
  return source.map((value, index) => ((value + delta[index]) % MOD + MOD) % MOD);
}

function modeParameter(rows) {
  let best = [Infinity, 0];
  for (let coefficient = 0; coefficient < MOD; coefficient += 1) {
    let errors = 0;
    for (const [feature, observed] of rows) if ((coefficient * feature) % MOD !== observed) errors += 1;
    if (errors < best[0]) best = [errors, coefficient];
  }
  return best[1];
}

function estimateMultipliers(cases) {
  const result = [];
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const rows = [];
    for (const [source, action, observed] of cases) {
      if (action[layer] && action.filter((x) => x !== 0).length === 1) {
        rows.push([action[layer], ((observed[layer] - source[layer]) % MOD + MOD) % MOD]);
      }
    }
    result.push(modeParameter(rows));
  }
  return result;
}

function fitFactorized(cases, graph, families, multipliers = null) {
  const fitted = multipliers ?? estimateMultipliers(cases);
  const [target, sources] = graph;
  const coefficients = [];
  for (let edge = 0; edge < 2; edge += 1) {
    const rows = [];
    for (const [source, action, observed] of cases) {
      const sourceIndex = sources[edge];
      if (action[sourceIndex] && action.filter((x) => x !== 0).length === 1) {
        const direct = fitted[target] * action[target];
        const residual = ((observed[target] - source[target] - direct) % MOD + MOD) % MOD;
        rows.push([headValue(families[edge], source, action, sourceIndex, target), residual]);
      }
    }
    coefficients.push(modeParameter(rows));
  }
  return { graph, families, multipliers: fitted, coefficients };
}

function nll(model, cases, noise = OOD_NOISE) {
  const matched = Math.log(1 - noise);
  const mismatched = Math.log(noise / (MOD - 1));
  let total = 0;
  let count = 0;
  for (const [source, action, observed] of cases) {
    const estimate = predict(model, source, action);
    for (let index = 0; index < LAYERS; index += 1) {
      total -= estimate[index] === observed[index] ? matched : mismatched;
      count += 1;
    }
  }
  return total / count;
}

function learn(train, selection) {
  const multipliers = estimateMultipliers(train);
  let best = null;
  for (const graph of GRAPHS) {
    for (const first of FAMILIES) {
      for (const second of FAMILIES) {
        const model = fitFactorized(train, graph, [first, second], multipliers);
        const loss = nll(model, selection, 0.08);
        if (best === null || loss < best.loss) best = { loss, model };
      }
    }
  }
  return best.model;
}

function same(left, right) { return JSON.stringify(left) === JSON.stringify(right); }
function close(left, right, tolerance = 1e-12) { return Math.abs(left - right) <= tolerance; }

const portable = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const published = JSON.parse(fs.readFileSync(analysisPath, "utf8"));
const ledger = JSON.parse(fs.readFileSync(ledgerPath, "utf8"));
const seed = Buffer.from(ledger.root_seed_hex_revealed_after_execution, "hex");
const commitment = crypto.createHash("sha256").update(seed).digest("hex");
if (commitment !== portable.root_seed_commitment) throw new Error("seed commitment mismatch");

const failures = [];
for (const world of portable.worlds) {
  const learned = learn(world.train, world.selection);
  const expected = world.expected_row;
  const trueModel = fitFactorized(world.train, world.graph, world.families);
  const graphIndex = GRAPHS.findIndex((graph) => same(graph, world.graph));
  const shifted = GRAPHS[(graphIndex + 11) % GRAPHS.length];
  const wrongModel = fitFactorized(world.train, shifted, world.families);
  const checks = [
    same(learned.graph, expected.learned_graph),
    same(learned.families, expected.learned_families),
    close(nll(learned, world.test), expected.metrics.learned_factorized.composition_nll),
    close(nll(trueModel, world.test), expected.metrics.correct_graph_correct_head.composition_nll),
    close(nll(wrongModel, world.test), expected.metrics.wrong_graph_correct_head.composition_nll),
  ];
  if (!checks.every(Boolean)) failures.push({ world_index: world.world_index, checks });
}

// Independently recompute the published world-level effect means from raw rows.
const effectNames = Object.keys(published.analysis.effect_intervals);
const rawEffects = published.analysis.world_effects;
const recomputedMeans = {};
for (const name of effectNames) {
  recomputedMeans[name] = rawEffects.reduce((sum, row) => sum + row[name], 0) / rawEffects.length;
  if (!close(recomputedMeans[name], published.analysis.effect_intervals[name].mean)) {
    failures.push({ effect: name, reason: "mean mismatch" });
  }
}

const audit = {
  status: "cleanroom_nodejs_reimplementation",
  language_runtime: process.version,
  project_imports: 0,
  world_count: portable.worlds.length,
  seed_commitment_verified: commitment === portable.root_seed_commitment,
  learned_factorized_worlds_reproduced: portable.worlds.length - failures.filter((x) => "world_index" in x).length,
  effect_means_reproduced: effectNames.length - failures.filter((x) => "effect" in x).length,
  failures,
  passed: failures.length === 0,
};
fs.writeFileSync(outputPath, `${JSON.stringify(audit, null, 2)}\n`);
console.log(JSON.stringify(audit, null, 2));
